"""Configuration layer.

Exposes the single typed `Settings` object that every other module should
import instead of touching `os.environ` directly. This keeps configuration
testable and makes it obvious where to look when something needs to change.
"""