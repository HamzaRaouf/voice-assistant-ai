import threading
import time
import logging
import psutil
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

@dataclass
class ThreadInfo:
    thread_id: int
    thread_name: str
    is_alive: bool
    is_daemon: bool
    start_time: Optional[float]
    cpu_percent: float
    memory_mb: float
    status: str

class ThreadMonitor:
    """Monitor and track thread information with detailed logging"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.thread_history: Dict[int, ThreadInfo] = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
    def get_current_threads(self) -> Dict[int, ThreadInfo]:
        """Get information about all current threads"""
        thread_info = {}
        
        try:
            # Get system thread information
            threads = self.process.threads()
            thread_dict = {t.id: t for t in threads}
            
            # Get Python threading information
            for thread in threading.enumerate():
                thread_id = thread.ident
                if thread_id is None:
                    continue
                    
                # Get system thread info if available
                sys_thread = thread_dict.get(thread_id)
                cpu_percent = 0.0
                
                try:
                    if sys_thread:
                        cpu_percent = getattr(sys_thread, 'cpu_percent', 0.0)
                except (AttributeError, psutil.NoSuchProcess):
                    cpu_percent = 0.0
                
                thread_info[thread_id] = ThreadInfo(
                    thread_id=thread_id,
                    thread_name=thread.name,
                    is_alive=thread.is_alive(),
                    is_daemon=thread.daemon,
                    start_time=getattr(thread, '_started', None),
                    cpu_percent=cpu_percent,
                    memory_mb=self.process.memory_info().rss / 1024 / 1024,
                    status="running" if thread.is_alive() else "stopped"
                )
                
        except Exception as e:
            logging.error(f"Error getting thread information: {e}")
            
        return thread_info
    
    def log_thread_cancellation(self, thread_id: int, thread_name: str, reason: str = "interrupted"):
        """Log when a thread is cancelled"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logging.info(f"{Fore.RED}[{timestamp}] Thread CANCELLED: {thread_name} (ID: {thread_id}) - Reason: {reason}{Fore.RESET}")
        
        # Update thread history
        with self._lock:
            if thread_id in self.thread_history:
                self.thread_history[thread_id].status = f"cancelled ({reason})"
    
    def log_thread_start(self, thread_id: int, thread_name: str, process_type: str = "unknown"):
        """Log when a thread starts"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logging.info(f"{Fore.GREEN}[{timestamp}] Thread STARTED: {thread_name} (ID: {thread_id}) - Type: {process_type}{Fore.RESET}")
    
    def log_thread_completion(self, thread_id: int, thread_name: str, duration: float):
        """Log when a thread completes successfully"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logging.info(f"{Fore.BLUE}[{timestamp}] Thread COMPLETED: {thread_name} (ID: {thread_id}) - Duration: {duration:.2f}s{Fore.RESET}")
        
        # Update thread history
        with self._lock:
            if thread_id in self.thread_history:
                self.thread_history[thread_id].status = "completed"
    
    def log_active_threads(self):
        """Log current active threads"""
        current_threads = self.get_current_threads()
        active_threads = {tid: info for tid, info in current_threads.items() if info.is_alive}
        
        if active_threads:
            logging.info(f"{Fore.CYAN}Active Threads Summary ({len(active_threads)} total):{Fore.RESET}")
            for tid, info in active_threads.items():
                status_color = Fore.GREEN if info.status == "running" else Fore.YELLOW
                daemon_str = " [DAEMON]" if info.is_daemon else ""
                logging.info(f"  {status_color}- {info.thread_name} (ID: {tid}){daemon_str} - CPU: {info.cpu_percent:.1f}%{Fore.RESET}")
        else:
            logging.info(f"{Fore.CYAN}No active threads{Fore.RESET}")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous thread monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            logging.info(f"{Fore.CYAN}Thread monitoring started (interval: {interval}s){Fore.RESET}")
            
            while self.monitoring_active:
                try:
                    with self._lock:
                        current_threads = self.get_current_threads()
                        
                        # Update history
                        for tid, info in current_threads.items():
                            self.thread_history[tid] = info
                        
                        # Check for high CPU usage
                        high_cpu_threads = [
                            info for info in current_threads.values() 
                            if info.cpu_percent > 80.0 and info.is_alive
                        ]
                        
                        if high_cpu_threads:
                            for thread in high_cpu_threads:
                                logging.warning(f"{Fore.YELLOW}High CPU usage detected: {thread.thread_name} (ID: {thread.thread_id}) - {thread.cpu_percent:.1f}%{Fore.RESET}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"Error in thread monitoring: {e}")
                    time.sleep(1)
            
            logging.info(f"{Fore.CYAN}Thread monitoring stopped{Fore.RESET}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, name="ThreadMonitor", daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop thread monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def get_thread_summary(self) -> Dict[str, int]:
        """Get a summary of thread statistics"""
        current_threads = self.get_current_threads()
        
        return {
            "total_threads": len(current_threads),
            "active_threads": sum(1 for t in current_threads.values() if t.is_alive),
            "daemon_threads": sum(1 for t in current_threads.values() if t.is_daemon),
            "non_daemon_threads": sum(1 for t in current_threads.values() if not t.is_daemon),
        }
    
    def log_thread_summary(self):
        """Log a detailed thread summary"""
        summary = self.get_thread_summary()
        current_threads = self.get_current_threads()
        
        logging.info(f"{Fore.CYAN}Thread Summary:{Fore.RESET}")
        logging.info(f"  Total: {summary['total_threads']}")
        logging.info(f"  Active: {summary['active_threads']}")
        logging.info(f"  Daemon: {summary['daemon_threads']}")
        logging.info(f"  Non-daemon: {summary['non_daemon_threads']}")
        
        # Group by thread type/name pattern
        thread_types = {}
        for info in current_threads.values():
            if info.is_alive:
                thread_type = info.thread_name.split('_')[0] if '_' in info.thread_name else info.thread_name
                thread_types[thread_type] = thread_types.get(thread_type, 0) + 1
        
        if thread_types:
            logging.info(f"  Thread types:")
            for thread_type, count in sorted(thread_types.items()):
                logging.info(f"    {thread_type}: {count}")
    
    def get_cancelled_threads(self, limit: int = 10) -> List[ThreadInfo]:
        """Get recently cancelled threads"""
        with self._lock:
            cancelled = [
                info for info in self.thread_history.values() 
                if "cancelled" in info.status
            ]
            return sorted(cancelled, key=lambda x: x.start_time or 0, reverse=True)[:limit]
    
    def log_cancellation_history(self, limit: int = 5):
        """Log recent thread cancellations"""
        cancelled = self.get_cancelled_threads(limit)
        
        if cancelled:
            logging.info(f"{Fore.YELLOW}Recent Thread Cancellations (last {len(cancelled)}):{Fore.RESET}")
            for info in cancelled:
                start_time = datetime.fromtimestamp(info.start_time).strftime("%H:%M:%S") if info.start_time else "unknown"
                logging.info(f"  - {info.thread_name} (ID: {info.thread_id}) at {start_time} - {info.status}")
        else:
            logging.info(f"{Fore.GREEN}No recent thread cancellations{Fore.RESET}")

# Global thread monitor instance
thread_monitor = ThreadMonitor()

# Decorator for monitoring thread lifecycle
def monitor_thread(process_type: str = "unknown"):
    """Decorator to monitor thread lifecycle"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            start_time = time.time()
            
            thread_monitor.log_thread_start(thread_id, thread_name, process_type)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                thread_monitor.log_thread_completion(thread_id, thread_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                thread_monitor.log_thread_cancellation(thread_id, thread_name, f"error: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Context manager for thread monitoring
class ThreadContext:
    """Context manager for monitoring thread operations"""
    
    def __init__(self, operation_name: str, process_type: str = "unknown"):
        self.operation_name = operation_name
        self.process_type = process_type
        self.thread_id = None
        self.thread_name = None
        self.start_time = None
    
    def __enter__(self):
        self.thread_id = threading.current_thread().ident
        self.thread_name = threading.current_thread().name
        self.start_time = time.time()
        
        thread_monitor.log_thread_start(self.thread_id, f"{self.thread_name}:{self.operation_name}", self.process_type)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            thread_monitor.log_thread_cancellation(
                self.thread_id, 
                f"{self.thread_name}:{self.operation_name}", 
                f"exception: {exc_type.__name__}"
            )
        else:
            thread_monitor.log_thread_completion(
                self.thread_id, 
                f"{self.thread_name}:{self.operation_name}", 
                duration
            )

# Utility functions
def start_thread_monitoring(interval: float = 5.0):
    """Start the global thread monitor"""
    thread_monitor.start_monitoring(interval)

def stop_thread_monitoring():
    """Stop the global thread monitor"""
    thread_monitor.stop_monitoring()

def log_current_threads():
    """Log current thread status"""
    thread_monitor.log_active_threads()

def log_thread_summary():
    """Log thread summary"""
    thread_monitor.log_thread_summary()

def log_cancellation_history():
    """Log recent cancellations"""
    thread_monitor.log_cancellation_history() 