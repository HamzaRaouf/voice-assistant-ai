import threading
import time
import logging
import queue
import signal
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

class ProcessType(Enum):
    LLM_INFERENCE = "llm_inference"
    TTS_GENERATION = "tts_generation"
    AUDIO_PLAYBACK = "audio_playback"

@dataclass
class ProcessInfo:
    thread_id: int
    process_type: ProcessType
    start_time: float
    status: str
    details: Dict[str, Any]

class CancellableProcess(ABC):
    """Abstract base class for cancellable processes"""
    
    def __init__(self, process_id: str, process_type: ProcessType):
        self.process_id = process_id
        self.process_type = process_type
        self.is_cancelled = threading.Event()
        self.is_completed = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.start_time = time.time()
        self.result = None
        self.error = None
        
    @abstractmethod
    def execute(self) -> Any:
        """Execute the process - to be implemented by subclasses"""
        pass
    
    def cancel(self):
        """Cancel the process"""
        logging.info(f"{Fore.YELLOW}Cancelling {self.process_type.value} process {self.process_id}{Fore.RESET}")
        self.is_cancelled.set()
        
    def is_running(self) -> bool:
        """Check if process is still running"""
        return self.thread and self.thread.is_alive() and not self.is_completed.is_set()
        
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for process completion"""
        return self.is_completed.wait(timeout)

class ProcessManager:
    """Manages all cancellable processes with thread tracking"""
    
    def __init__(self):
        self.active_processes: Dict[str, CancellableProcess] = {}
        self.process_history: Dict[str, ProcessInfo] = {}
        self._lock = threading.Lock()
        self._process_counter = 0
        
    def generate_process_id(self, process_type: ProcessType) -> str:
        """Generate a unique process ID"""
        with self._lock:
            self._process_counter += 1
            return f"{process_type.value}_{self._process_counter}_{int(time.time())}"
    
    def start_process(self, process: CancellableProcess) -> str:
        """Start a cancellable process"""
        with self._lock:
            # Cancel any existing processes of the same type
            self._cancel_processes_by_type(process.process_type)
            
            # Start new process
            process.thread = threading.Thread(
                target=self._run_process_wrapper, 
                args=(process,),
                name=f"{process.process_type.value}_{process.process_id}"
            )
            process.thread.daemon = True
            process.thread.start()
            
            self.active_processes[process.process_id] = process
            
            # Log process start
            thread_id = process.thread.ident
            logging.info(f"{Fore.GREEN}Started {process.process_type.value} process {process.process_id} on thread {thread_id}{Fore.RESET}")
            
            # Store process info
            self.process_history[process.process_id] = ProcessInfo(
                thread_id=thread_id,
                process_type=process.process_type,
                start_time=process.start_time,
                status="running",
                details={"process_id": process.process_id}
            )
            
            return process.process_id
    
    def _run_process_wrapper(self, process: CancellableProcess):
        """Wrapper to run process with error handling"""
        try:
            process.result = process.execute()
            if not process.is_cancelled.is_set():
                logging.info(f"{Fore.GREEN}Completed {process.process_type.value} process {process.process_id}{Fore.RESET}")
                self.process_history[process.process_id].status = "completed"
        except Exception as e:
            process.error = e
            logging.error(f"{Fore.RED}Error in {process.process_type.value} process {process.process_id}: {e}{Fore.RESET}")
            self.process_history[process.process_id].status = "error"
        finally:
            process.is_completed.set()
            with self._lock:
                if process.process_id in self.active_processes:
                    del self.active_processes[process.process_id]
    
    def cancel_process(self, process_id: str) -> bool:
        """Cancel a specific process"""
        with self._lock:
            if process_id in self.active_processes:
                process = self.active_processes[process_id]
                process.cancel()
                
                # Update process info
                if process_id in self.process_history:
                    self.process_history[process_id].status = "cancelled"
                
                thread_id = process.thread.ident if process.thread else "unknown"
                logging.info(f"{Fore.YELLOW}Cancelled process {process_id} on thread {thread_id}{Fore.RESET}")
                return True
        return False
    
    def cancel_all_processes(self):
        """Cancel all active processes"""
        with self._lock:
            process_ids = list(self.active_processes.keys())
            for process_id in process_ids:
                self.cancel_process(process_id)
    
    def _cancel_processes_by_type(self, process_type: ProcessType):
        """Cancel all processes of a specific type"""
        process_ids = [
            pid for pid, process in self.active_processes.items() 
            if process.process_type == process_type
        ]
        for process_id in process_ids:
            self.cancel_process(process_id)
    
    def get_active_processes(self) -> Dict[str, ProcessInfo]:
        """Get information about all active processes"""
        with self._lock:
            return {
                pid: ProcessInfo(
                    thread_id=process.thread.ident if process.thread else 0,
                    process_type=process.process_type,
                    start_time=process.start_time,
                    status="running",
                    details={"process_id": pid}
                )
                for pid, process in self.active_processes.items()
            }
    
    def get_process_history(self, limit: int = 10) -> Dict[str, ProcessInfo]:
        """Get recent process history"""
        sorted_history = dict(sorted(
            self.process_history.items(), 
            key=lambda x: x[1].start_time, 
            reverse=True
        )[:limit])
        return sorted_history
    
    def log_process_status(self):
        """Log current process status"""
        active = self.get_active_processes()
        if active:
            logging.info(f"{Fore.CYAN}Active processes: {len(active)}{Fore.RESET}")
            for pid, info in active.items():
                logging.info(f"  - {info.process_type.value} (PID: {pid}, Thread: {info.thread_id})")
        else:
            logging.info(f"{Fore.CYAN}No active processes{Fore.RESET}")

# Global process manager instance
process_manager = ProcessManager()

class InterruptibleLLMProcess(CancellableProcess):
    """Cancellable LLM inference process"""
    
    def __init__(self, process_id: str, llm_function: Callable, *args, **kwargs):
        super().__init__(process_id, ProcessType.LLM_INFERENCE)
        self.llm_function = llm_function
        self.args = args
        self.kwargs = kwargs
    
    def execute(self) -> Any:
        """Execute LLM inference with cancellation checks"""
        # For Ollama, we need to implement streaming with cancellation checks
        try:
            # Check cancellation before starting
            if self.is_cancelled.is_set():
                logging.info(f"{Fore.YELLOW}LLM process {self.process_id} cancelled before execution{Fore.RESET}")
                return None
            
            # Execute the LLM function
            # We'll need to modify this based on the specific LLM implementation
            result = self.llm_function(*self.args, **self.kwargs)
            
            # Check cancellation after completion
            if self.is_cancelled.is_set():
                logging.info(f"{Fore.YELLOW}LLM process {self.process_id} cancelled after execution{Fore.RESET}")
                return None
                
            return result
            
        except Exception as e:
            if self.is_cancelled.is_set():
                logging.info(f"{Fore.YELLOW}LLM process {self.process_id} cancelled due to interruption{Fore.RESET}")
                return None
            raise e

class InterruptibleTTSProcess(CancellableProcess):
    """Cancellable TTS generation process"""
    
    def __init__(self, process_id: str, tts_function: Callable, *args, **kwargs):
        super().__init__(process_id, ProcessType.TTS_GENERATION)
        self.tts_function = tts_function
        self.args = args
        self.kwargs = kwargs
    
    def execute(self) -> Any:
        """Execute TTS generation with cancellation checks"""
        try:
            # Check cancellation before starting
            if self.is_cancelled.is_set():
                logging.info(f"{Fore.YELLOW}TTS process {self.process_id} cancelled before execution{Fore.RESET}")
                return None
            
            # Execute the TTS function
            result = self.tts_function(*self.args, **self.kwargs)
            
            # Check cancellation after completion
            if self.is_cancelled.is_set():
                logging.info(f"{Fore.YELLOW}TTS process {self.process_id} cancelled after execution{Fore.RESET}")
                return None
                
            return result
            
        except Exception as e:
            if self.is_cancelled.is_set():
                logging.info(f"{Fore.YELLOW}TTS process {self.process_id} cancelled due to interruption{Fore.RESET}")
                return None
            raise e

# Convenience functions for process management
def start_llm_process(llm_function: Callable, *args, **kwargs) -> str:
    """Start a cancellable LLM process"""
    process_id = process_manager.generate_process_id(ProcessType.LLM_INFERENCE)
    process = InterruptibleLLMProcess(process_id, llm_function, *args, **kwargs)
    return process_manager.start_process(process)

def start_tts_process(tts_function: Callable, *args, **kwargs) -> str:
    """Start a cancellable TTS process"""
    process_id = process_manager.generate_process_id(ProcessType.TTS_GENERATION)
    process = InterruptibleTTSProcess(process_id, tts_function, *args, **kwargs)
    return process_manager.start_process(process)

def cancel_all_ai_processes():
    """Cancel all LLM and TTS processes"""
    logging.info(f"{Fore.YELLOW}Cancelling all AI processes due to speech interruption{Fore.RESET}")
    process_manager.cancel_all_processes()
    process_manager.log_process_status() 