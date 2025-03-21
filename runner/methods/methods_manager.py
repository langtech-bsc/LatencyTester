from abc import ABC, abstractmethod
import os
import sys
import importlib.util
import logging
from typing import Optional, Type, Union

logger = logging.getLogger(__name__)


class BaseTest(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def invoke(self):
        """Function to be implemented by subclasses"""
        pass


class MethodManager:
    registered_tests: dict[str, Type[BaseTest]] = {}

    @classmethod
    def get_method(cls, name: str) -> Type[BaseTest]:
        """Retrieve a test class by name."""
        if name in cls.registered_tests:
            return cls.registered_tests[name]
        raise KeyError(f"Test '{name}' not found in registered tests.")

    @classmethod
    def _register_method(cls, test_class: Type[BaseTest], test_name: Optional[Union[str, list[str]]] = None, force: bool = True) -> None:
        """Internal method to register a test class."""
        if not issubclass(test_class, BaseTest):
            raise TypeError(f"Module must be a subclass of BaseTest, but got {type(test_class)}")

        test_names = [test_class.__name__] if test_name is None else ([test_name] if isinstance(test_name, str) else test_name)

        for name in test_names:
            if not force and name in cls.registered_tests:
                raise KeyError(f"{name} is already registered at {cls.registered_tests[name].__module__}")
            cls.registered_tests[name] = test_class

    @classmethod
    def register_method(cls, name: Optional[Union[str, list[str]]] = None, force: bool = True, test_class: Union[Type[BaseTest], None] = None) -> Union[Type[BaseTest], callable]:
        """Decorator or function to register a test class."""
        if test_class is not None:
            cls._register_method(test_class=test_class, test_name=name, force=force)
            return test_class

        def _register(test_cls):
            cls._register_method(test_class=test_cls, test_name=name, force=force)
            return test_cls

        return _register

    @classmethod
    def import_method(cls, plugin_path: str) -> None:
        """Dynamically import a test class from a file path."""
        if not os.path.isfile(plugin_path):
            raise FileNotFoundError(f"File not found: {plugin_path}")

        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(spec)

        sys.modules[module_name] = module  # Register module globally
        try:
            spec.loader.exec_module(module)
            logger.info(f"Successfully loaded module '{module_name}' from {plugin_path}")
        except Exception as e:
            logger.error(f"Failed to load module '{module_name}' from {plugin_path}: {e}")
