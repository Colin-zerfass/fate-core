from manim import *


Text.set_default(color = BLACK)
Arrow.set_default(color = BLACK)
class CreateArrow(Scene):
    def construct(self):
        def CreateDiagram(start,len1, len2):
            self.camera.background_color = "#FFFFFF"
            self.camera.set_color = '#000000'
            a1 = start + len1
            a2 = start + len2
            arrow1 = Arrow(start, a1, buff = 0)  # makes Arrow
            arrow2 = Arrow(start, a2, buff = 0)  # makes Arrow
            dFADtext = Text("dFAD").next_to(arrow1.get_center(), buff = 0.5)
            modeltext = Text("Model").next_to(arrow2.get_center(), LEFT, buff = 0.2)
            arrowdif = Arrow(arrow2.get_end(), arrow1.get_end())
            if arrowdif.get_length() <1 :
                color = "#139E36"
            else: 
                color = "#CB290C"
            arrowdif.color = color
            self.add(arrow1, arrow2 ,dFADtext, modeltext,arrowdif)  # show the circle on screen
        CreateDiagram(np.array([4,0,0]), np.array([2,1,0]), [-1,2.5,0])
        CreateDiagram(np.array([-3,0,0]), np.array([2,1,0]), [1,2,0])
        diftext = Text("dFAD - Model").to_edge(UP)
        self.add(diftext)


        