import uuid


def get_request_id() -> str:
    return str(uuid.uuid4.hex())