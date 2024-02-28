from contextlib import closing
from inputimeout import inputimeout, TimeoutOccurred
import socket
import sys
import time

## Functions

def prompt_yes_or_no(query, timed=False, timeout=30, default_response=None):
    while True:
        if timed: # inputimeout does NOT work when called from a subprocess!
            assert default_response is not None
            try:
                response = inputimeout(prompt=query + ' (timeout in %ds) [Y/n] '%(timeout), timeout=timeout).lower()
            except TimeoutOccurred:
                response = default_response
        else:
            if default_response is None:
                response = input(query + ' [Y/n] ').lower()
            else:
                response = default_response 
        if response in {'y', 'yes'}:
            return True
        elif response in {'n', 'no'}:
            return False
        else:
            print('Invalid response!\n')

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

## Classes

class ROSRate(object):
    '''
    http://docs.ros.org/diamondback/api/rostime/html/rate_8cpp_source.html
    '''
    def __init__(self, frequency):
        assert frequency > 0, 'Frequency must be greated than zero!'
        self._freq = frequency
        self._start = time.time()
        self._actual_cycle_time = 1/self._freq

    def reset(self):
        self._start = time.time()
    
    def sleep(self):
        expected_end = self._start + 1/self._freq
        actual_end = time.time()

        if actual_end < self._start: # detect backward jumps in time
            expected_end = actual_end + 1/self._freq

        # calculate sleep time
        sleep_duration = expected_end - actual_end
        # set the actual amount of time the loop took in case the user wants to know
        self._actual_cycle_time = actual_end - self._start

        # reset start time
        self._start = expected_end

        if sleep_duration <= 0:
            # if we've jumped forward in time, or the loop has taken more than a full extra cycle, reset our cycle
            if actual_end > expected_end + 1/self._freq:
                self._start = actual_end
            return True

        return time.sleep(sleep_duration)