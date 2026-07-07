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

from __future__ import annotations

from typing import Callable, TypeVar

from .interface import FilterPlugin, FlowControlPlugin, PickerPlugin, ProfileHandler, ScorerPlugin

T = TypeVar("T")

# Internal registry dictionaries mapping string names to Plugin classes
_SCORERS: dict[str, type[ScorerPlugin]] = {}
_PICKERS: dict[str, type[PickerPlugin]] = {}
_FILTERS: dict[str, type[FilterPlugin]] = {}
_PROFILE_HANDLERS: dict[str, type[ProfileHandler]] = {}
_FLOW_CONTROLS: dict[str, type[FlowControlPlugin]] = {}


def register_scorer(name: str) -> Callable[[type[ScorerPlugin]], type[ScorerPlugin]]:
    """Decorator to register a custom Scorer class under a specific string name."""

    def wrapper(cls: type[ScorerPlugin]) -> type[ScorerPlugin]:
        _SCORERS[name] = cls
        return cls

    return wrapper


def register_picker(name: str) -> Callable[[type[PickerPlugin]], type[PickerPlugin]]:
    """Decorator to register a custom Picker class under a specific string name."""

    def wrapper(cls: type[PickerPlugin]) -> type[PickerPlugin]:
        _PICKERS[name] = cls
        return cls

    return wrapper


def register_filter(name: str) -> Callable[[type[FilterPlugin]], type[FilterPlugin]]:
    """Decorator to register a custom Filter class under a specific string name."""

    def wrapper(cls: type[FilterPlugin]) -> type[FilterPlugin]:
        _FILTERS[name] = cls
        return cls

    return wrapper


def register_profile_handler(
    name: str,
) -> Callable[[type[ProfileHandler]], type[ProfileHandler]]:
    """Decorator to register a custom ProfileHandler class under a specific string name."""

    def wrapper(cls: type[ProfileHandler]) -> type[ProfileHandler]:
        _PROFILE_HANDLERS[name] = cls
        return cls

    return wrapper


def register_flow_control(
    name: str,
) -> Callable[[type[FlowControlPlugin]], type[FlowControlPlugin]]:
    """Decorator to register a custom FlowControl class under a specific string name."""

    def wrapper(cls: type[FlowControlPlugin]) -> type[FlowControlPlugin]:
        _FLOW_CONTROLS[name] = cls
        return cls

    return wrapper


def build_plugin(registry: dict[str, type[T]], type_name: str, **kwargs: object) -> T:
    """Helper method to instantiate a class from a specific registry category by its string name."""
    cls = registry.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown plugin type '{type_name}'. Available types: {list(registry.keys())}"
        )

    # Instantiate the plugin with provided configurations/kwargs
    return cls(**kwargs)


def build_scorer(type_name: str, **kwargs: object) -> ScorerPlugin:
    """Instantiate a registered Scorer class based on its string identifier and config."""
    return build_plugin(_SCORERS, type_name, **kwargs)


def build_picker(type_name: str, **kwargs: object) -> PickerPlugin:
    """Instantiate a registered Picker class based on its string identifier and config."""
    return build_plugin(_PICKERS, type_name, **kwargs)


def build_filter(type_name: str, **kwargs: object) -> FilterPlugin:
    """Instantiate a registered Filter class based on its string identifier and config."""
    return build_plugin(_FILTERS, type_name, **kwargs)


def build_profile_handler(type_name: str, **kwargs: object) -> ProfileHandler:
    """Instantiate a registered ProfileHandler class based on its string identifier and config."""
    return build_plugin(_PROFILE_HANDLERS, type_name, **kwargs)


def build_flow_control(type_name: str, **kwargs: object) -> FlowControlPlugin:
    """Instantiate a registered FlowControl class based on its string identifier and config."""
    return build_plugin(_FLOW_CONTROLS, type_name, **kwargs)
