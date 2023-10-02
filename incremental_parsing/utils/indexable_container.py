from typing import Protocol, TypeVar

ReturnType = TypeVar("ReturnType", covariant=True)


class IndexableContainer(Protocol[ReturnType]):
    def __getitem__(self, key: int) -> ReturnType:
        ...

    def __len__(self) -> int:
        ...
