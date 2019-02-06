class RecognizedNumber:

        def __init__(self, coordinates):
            self.__x = coordinates[0]
            self.__y = coordinates[1]
            self.__w = coordinates[2]
            self.__h = coordinates[3]
            self.__bluePassed = False
            self.__greenPassed = False
            self.__prediction = -1
            self.__predicted = False
            self.__considered = True

        def update_coordinates(self, coordinates):
            self.__x = coordinates[0]
            self.__y = coordinates[1]
            self.__w = coordinates[2]
            self.__h = coordinates[3]

        def get_considered(self):
            return self.__considered
        
        def set_considered(self, value):
            self.__considered = value

        def get_predicted(self):
            return self.__predicted
        
        def set_predicted(self, value):
            self.__predicted = value

        def get_prediction(self):
            return self.__prediction

        def set_prediction(self, value):
            self.__prediction = value

        def get_x(self):
            return self.__x

        def get_y(self):
            return self.__y

        def get_center(self):
            return self.__x + self.__w/2, self.__y + self.__h/2

        def get_top_left(self):
            return (self.__x, self.__y)

        def get_top_right(self):
            return (self.__x + self.__w, self.__y)

        def get_bottom_left(self):
            return (self.__x, self.__y + self.__h)

        def get_bottom_right(self):
            return (self.__x + self.__w, self.__y + self.__h)

        def get_bluePassed(self):
            return self.__bluePassed

        def set_bluePassed(self, value):
            self.__bluePassed = value

        def get_greenPassed(self):
            return self.__greenPassed

        def set_greenPassed(self, value):
            self.__greenPassed = value

        