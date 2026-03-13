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

from typing import Callable, Type, Dict, Any, TypeVar
from .interface import ScorerPlugin, PickerPlugin, FilterPlugin, ProfileHandler

T = TypeVar("T")

# Internal registry dictionaries mapping string names to Plugin classes
_SCORERS: Dict[str, Type[ScorerPlugin]] = {}
_PICKERS: Dict[str, Type[PickerPlugin]] = {}
_FILTERS: Dict[str, Type[FilterPlugin]] = {}
_PROFILE_HANDLERS: Dict[str, Type[ProfileHandler]] = {}


def register_scorer(name: str) -> Callable[[Type[ScorerPlugin]], Type[ScorerPlugin]]:
    """Decorator to register a custom Scorer class under a specific string name."""
    def wrapper(cls: Type[ScorerPlugin]) -> Type[ScorerPlugin]:
        _SCORERS[name] = cls
        return cls
    return wrapper


def register_picker(name: str) -> Callable[[Type[PickerPlugin]], Type[PickerPlugin]]:
    """Decorator to register a custom Picker class under a specific string name."""
    def wrapper(cls: Type[PickerPlugin]) -> Type[PickerPlugin]:
        _PICKERS[name] = cls
        return cls
    return wrapper


def register_filter(name: str) -> Callable[[Type[FilterPlugin]], Type[FilterPlugin]]:
    """Decorator to register a custom Filter class under a specific string name."""
    def wrapper(cls: Type[FilterPlugin]) -> Type[FilterPlugin]:
        _FILTERS[name] = cls
        return cls
    return wrapper


def register_profile_handler(name: str) -> Callable[[Type[ProfileHandler]], Type[ProfileHandler]]:
    """Decorator to register a custom ProfileHandler class under a specific string name."""
    def wrapper(cls: Type[ProfileHandler]) -> Type[ProfileHandler]:
        _PROFILE_HANDLERS[name] = cls
        return cls
    return wrapper


def build_plugin(registry: Dict[str, Type[T]], type_name: str, **kwargs: Any) -> T:
    """Helper method to instantiate a class from a specific registry category by its string name."""
    cls = registry.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown plugin type '{type_name}'. Available types: {list(registry.keys())}")
    
    # Instantiate the plugin with provided configurations/kwargs
    return cls(**kwargs)


def build_scorer(type_name: str, **kwargs: Any) -> ScorerPlugin:
    """Instantiate a registered Scorer class based on its string identifier and config."""
    return build_plugin(_SCORERS, type_name, **kwargs)


def build_picker(type_name: str, **kwargs: Any) -> PickerPlugin:
    """Instantiate a registered Picker class based on its string identifier and config."""
    return build_plugin(_PICKERS, type_name, **kwargs)


def build_filter(type_name: str, **kwargs: Any) -> FilterPlugin:
    """Instantiate a registered Filter class based on its string identifier and config."""
    return build_plugin(_FILTERS, type_name, **kwargs)


def build_profile_handler(type_name: str, **kwargs: Any) -> ProfileHandler:
    """Instantiate a registered ProfileHandler class based on its string identifier and config."""
    return build_plugin(_PROFILE_HANDLERS, type_name, **kwargs)

