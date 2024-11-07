from pydantic import BaseModel, ConfigDict


class FrigateBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=()) # forbid
