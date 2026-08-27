from enum import Enum, auto

class PlatformMode(Enum): 
    OFF = auto()       # disabled/safe, no search, no tracking
    SEARCHING = auto() # enabled, but no valid object yet
    PRE_SLEWING_TO_LEAD = auto() # pre aiming probably, for TENTATIVE tracks
    SLEWING_TO_LEAD = auto()
    FOLLOWING_LEAD = auto()