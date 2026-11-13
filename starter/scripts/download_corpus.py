"""Download the manifest corpus and verify every file checksum."""
import hashlib
import json
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for document in manifest["documents"]:
        destination = REPO_ROOT / document["source_path_or_url"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = document["sha256"]

        if destination.exists() and sha256(destination) == expected:
            print(f"OK (cached): {destination}")
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading: {document['download_url']}")
        urllib.request.urlretrieve(document["download_url"], temporary)

        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: "
                f"expected={expected}, actual={actual}"
            )

        temporary.replace(destination)
        print(f"Verified: {destination}")


if __name__ == "__main__":
    main()
