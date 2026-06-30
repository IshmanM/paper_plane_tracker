from enum import Enum, auto

class PlatformMode(Enum): 
    OFF = auto()       # disabled/safe, no search, no tracking
    SEARCHING = auto() # enabled, but no valid object yet
    SLEWING_TO_LEAD = auto()
    FOLLOWING_LEAD = auto()