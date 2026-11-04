"""Parse the documented containerd 1.7 `plugins list --detailed` output."""

from pathlib import PurePosixPath


def overlayfs_roots(text):
    plugins = []
    current = None
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Type:"):
            current = {
                "type": line.split(":", 1)[1].strip(),
                "roots": [],
                "error": False,
            }
            plugins.append(current)
            section = None
        elif current is not None:
            if line.startswith("ID:"):
                current["id"] = line.split(":", 1)[1].strip()
            elif line.endswith(":"):
                section = line[:-1]
                if section == "Error":
                    current["error"] = True
            elif section == "Exports":
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0] == "root":
                    current["roots"].append(parts[1])
    matching = [
        p
        for p in plugins
        if p["type"] == "io.containerd.snapshotter.v1" and p.get("id") == "overlayfs"
    ]
    assert (
        len(matching) == 1 and not matching[0]["error"]
    ), "overlayfs plugin unavailable"
    roots = matching[0]["roots"]
    assert (
        len(roots) == 1
        and roots[0].startswith("/")
        and ".." not in PurePosixPath(roots[0]).parts
    ), "invalid or ambiguous overlayfs root"
    return roots
