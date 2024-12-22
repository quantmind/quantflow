from pathlib import Path


class Vault:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.touch(exist_ok=True)
        self.data = self.load()

    def load(self) -> dict[str, str]:
        data = {}
        with open(self.path) as file:
            for line in file:
                key, value = line.strip().split("=")
                data[key] = value
        return data

    def add(self, key: str, value: str) -> None:
        self.data[key] = value
        self.save()

    def delete(self, key: str) -> bool:
        if self.data.pop(key, None) is not None:
            self.save()
            return True
        return False

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def keys(self) -> list[str]:
        return sorted(self.data)

    def save(self) -> None:
        with open(self.path, "w") as file:
            for key in sorted(self.data):
                value = self.data[key]
                file.write(f"{key}={value}\n")
