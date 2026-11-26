from pydantic import BaseModel

ALL_REGIONS = ("NA", "EMEA", "APAC", "LATAM")


class UserContext(BaseModel):
    """Who is asking. Every tool enforces these permissions itself, never the prompt."""

    user_id: str
    display_name: str
    groups: list[str]
    region_scope: list[str] | None = None  # None means every region

    @property
    def is_region_restricted(self) -> bool:
        return self.region_scope is not None

    def region_scope_setting(self) -> str:
        """Value for the Postgres `app.region_scope` setting used by row-level security."""
        return "*" if self.region_scope is None else ",".join(self.region_scope)

    def can_read(self, acl: list[str]) -> bool:
        return "all" in acl or bool(set(acl) & set(self.groups))


DEMO_USERS = {
    "finance.admin": UserContext(
        user_id="finance.admin",
        display_name="Fiona Park (Finance)",
        groups=["finance", "leadership"],
    ),
    "emea.sales": UserContext(
        user_id="emea.sales",
        display_name="Marco Rossi (EMEA Sales Manager)",
        groups=["sales"],
        region_scope=["EMEA"],
    ),
    "employee": UserContext(
        user_id="employee",
        display_name="Sam Lee (Employee)",
        groups=[],
    ),
}
