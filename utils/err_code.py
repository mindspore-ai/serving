from enum import IntEnum, unique


@unique
class AgentStatus(IntEnum):
    free = 0x1
    busy = 0x2
    connected = 0x4
    unconnected = 0x8
