import time
import os
from unittest.mock import patch

class MockSSD:
    """
    Test Double that intercepts os.pread to simulate physical SSD behavior.
    Allows injecting latency, tracking byte reads, and testing race conditions.
    """
    def __init__(self, bandwidth_gb_s: float, latency_ms: float = 0.0):
        self.bw_bytes_s = bandwidth_gb_s * 1024**3
        self.latency_s = latency_ms / 1000.0
        self.total_bytes_read = 0
        self.read_history = []
        self._original_pread = os.pread

    def mock_pread(self, fd, length, offset):
        """Simulates physical SSD read characteristics."""
        # 1. Simulate NVMe seek latency
        if self.latency_s > 0:
            time.sleep(self.latency_s)
            
        # 2. Simulate bandwidth limits
        if self.bw_bytes_s > 0:
            transfer_time = length / self.bw_bytes_s
            time.sleep(transfer_time)
            
        self.total_bytes_read += length
        
        # We can't reliably get the filename from the raw file descriptor in a test,
        # but we can store the tuple.
        self.read_history.append((fd, offset, length))
        
        # 3. Actually read the data so MLX doesn't crash on garbage
        return self._original_pread(fd, length, offset)
        
    def attach(self):
        return patch('os.pread', side_effect=self.mock_pread)
