# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import threading
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class InflightStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._inflight: Dict[str, int] = {}

    def increment(self, endpoint_name: str) -> None:
        """Called proactively in `_choose_server` the instant a route is decided (PreRequest)."""
        with self._lock:
            self._inflight[endpoint_name] = self._inflight.get(endpoint_name, 0) + 1

    def decrement(self, endpoint_name: str) -> None:
        """Called in `generate` the instant the physical GPU completes the request (ResponseComplete)."""
        with self._lock:
            if self._inflight.get(endpoint_name, 0) > 0:
                self._inflight[endpoint_name] -= 1
            else:
                logger.warning(f"Attempted to decrement inflight store for {endpoint_name} but it is already 0.")

    def get(self, endpoint_name: str) -> int:
        """Retrieves the active, uncompleted request count for this endpoint."""
        with self._lock:
             return self._inflight.get(endpoint_name, 0)

    def get_all(self) -> Dict[str, int]:
        """Provides a snapshot of the entire active cluster load."""
        with self._lock:
            return self._inflight.copy()
