# Default Objects, Gridworld consist of
class GridworldObject:

    def __init__(self):
        pass

    def walkable(self):
        return False

    def end(self):
        return False

    def get_name(self):
        return self.__class__.__name__


class Wall(GridworldObject):
    pass


class Floor(GridworldObject):
    def walkable(self):
        return True


class Lava(GridworldObject):
    def walkable(self):
        return True

    def end(self):
        return True


class Goal(GridworldObject):
    def walkable(self):
        return True

    def end(self):
        return True


class Checkpoint(GridworldObject):
    def walkable(self):
        return True

    def __init__(self, action_ind):
        self.action_ind = action_ind
        super().__init__()
