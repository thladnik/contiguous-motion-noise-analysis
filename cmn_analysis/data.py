# TODO: add alterantive, simplified data model for use without caload


class Entity(dict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class Animal(Entity):

    @property
    def recordings(self):
        return self['__recordings']


class Recording(Entity):

    @property
    def animal(self):
        return self['__animal']

    def phases(self):
        return self['__phases']

    @property
    def rois(self):
        return self['__rois']


class Roi(Entity):

    @property
    def animal(self):
        return self.recording['__animal']

    @property
    def recording(self):
        return self['__recording']


class Phase(Entity):

    @property
    def animal(self):
        return self.recording['__animal']

    @property
    def recording(self):
        return self['__recording']
