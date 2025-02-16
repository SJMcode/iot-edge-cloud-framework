from dataclasses import dataclass, field
from typing import Optional, Any, Union, Literal
from uuid import UUID
from datetime import datetime
from tree_cli.utils import parse_date
from json import dumps as json_dump

NodeType = Literal["Node", "Root", "Bool", "List", "Str", "Number", "Unknown"]


@dataclass
class Error(Exception):
    name: str
    error: Optional[Any] = field(repr=False, default=None)

    def __str__(self) -> str:
        if self.error is None:
            return f"<{self.name}>"
        return f"<{self.name}({self.error})>"


class User:
    id: UUID
    name: str
    creation_date: datetime
    write_allowed: bool
    update_at: Optional[datetime]

    def __init__(
        self,
        id: Union[UUID, str],
        name: str,
        creationDate: Union[datetime, str],
        writeAllowed: bool,
        updateDate: Union[datetime, str] = None,
    ):
        self.name = name
        self.write_allowed = writeAllowed

        if isinstance(id, str):
            self.id = UUID(id)
        else:
            self.id = id

        if isinstance(creationDate, str):
            self.creation_date = parse_date(creationDate)
        else:
            self.creation_date = creationDate

        if updateDate is None:
            self.update_at = None
        elif isinstance(updateDate, str):
            if updateDate == "0001-01-01T00:00:00Z":
                self.update_at = None
            else:
                self.update_at = parse_date(updateDate)
        else:
            self.update_at = updateDate

    def __str__(self) -> str:
        return "\n".join(
            [
                "User:",
                f"\tID: {repr(self.id)}",
                f"\tUser Name: {self.name}",
                f'\tCan Write: {"Yes" if self.write_allowed else "No"}',
                f"\tCreation Date: {self.creation_date}",
                f'\tUpdate Date: {self.update_at or "No update"}',
            ]
        )


class Node:
    id: UUID
    parent: Optional[UUID]
    name: str
    creation_date: datetime
    type: NodeType
    value: Any

    def __init__(
        self,
        id: Union[str, UUID],
        name: str,
        creationDate: Union[datetime, str],
        type: NodeType,
        value: Any,
        parent: Optional[Union[str, UUID]] = None,
    ):
        self.name = name
        self.type = type
        self.value = value

        if isinstance(id, str):
            self.id = UUID(id)
        else:
            self.id = id

        if parent is not None:
            if isinstance(parent, str):
                self.parent = UUID(parent)
            else:
                self.parent = parent

        if isinstance(creationDate, str):
            self.creation_date = parse_date(creationDate)
        else:
            self.creation_date = creationDate

    @property
    def json_value(self) -> str:
        if self.type in ["Node", "Root"]:
            return json_dump([repr(val) for val in self.value], indent="\t")

        return json_dump(self.value)

    def __str__(self) -> str:
        lines = ["Node:", f"\tID: {repr(self.id)}"]

        if self.parent is not None:
            lines.append(f"\t Parent: {repr(self.id)}")

        lines += [
            f"\tName: {self.name}",
            f"\tCreation Date: {self.creation_date}",
            f"\tType: {self.type}",
        ]

        return "\n".join(lines)


class Root(Node):
    def __str__(self) -> str:
        return "\n".join(
            [
                "Root:",
                f"\tID: {repr(self.id)}",
                f"\tName: {self.name}",
                f"\tCreation Date: {self.creation_date}",
                f"\tChilds: {self.json_value}",
            ]
        )
