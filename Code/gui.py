from Tkinter import *
from tkFileDialog import askopenfilename #
from PIL import ImageTk, Image

class dropDownMenu:
    label3 = None
    def __init__(self,master):
        self.mymaster = master
        self.myMenu = Menu(master)
        master.config(menu=self.myMenu)

        self.subMenu = Menu(self.myMenu)
        self.myMenu.add_cascade(label="File", menu=self.subMenu)
        self.subMenu.add_command(label="Open File...", command=self.browsePicture)
        self.subMenu.add_separator()

        self.helpMenu = Menu(self.myMenu)
        self.myMenu.add_cascade(label="Help", menu=self.helpMenu)
        self.helpMenu.add_command(label="About...", command= self.clickAbout)
        self.helpMenu.add_separator()

    def doNothing(self):
        print(".....")

    def clickAbout(self):

        ABOUT_TEXT = """About this GUI."""

        toplevel = Toplevel()
        toplevel.wm_title("About")

        label1 = Label(toplevel, text=ABOUT_TEXT, height=0, width=0)

        label1.pack()

    def browsePicture(self):
        Tk().withdraw()
        self.mymaster.ImageTitle = askopenfilename()
        self.img = ImageTk.PhotoImage(Image.open(self.mymaster.ImageTitle))
        self.mymaster.InputImagePanel.config(image = self.img)
        self.mymaster.update()


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.mymaster = master
        self.pack()
        self.createWidgets()

    def process(self):
        import learner
        self.img = Image.open(self.mymaster.ImageTitle)
        myGraph = learner.readImage(self.img)

        import networkx as nx
        import matplotlib.pyplot as plt

        for node in myGraph.nodes():
            xPos = node.x
            yPos = node.y
            #TODO: Draw node at (xPos, yPos)
            
        for edge in myGraph.edges():
            x0 = edge[0].x
            y0 = edge[0].y
            x1 = edge[1].x
            y1 = edge[1].y
            #TODO: Draw line from (x0, y0) to (x1, y1)
        #nx.draw(myGraph, node_color='k', node_size=25)
        plt.savefig(self.mymaster.ImageTitle+"-Output.png")
        plt.clf()

        self.img = ImageTk.PhotoImage(Image.open(self.mymaster.ImageTitle+"-Output.png"))
        self.mymaster.OutputImagePanel.config(image = self.img)
        self.mymaster.update()

    def createWidgets(self):

        self.process = Button(self, text = "process", command = self.process)
        self.process.pack(side="top",fill = "both", expand = "yes", padx=15, pady=15)

        # self.QUIT = Button(self, text = "QUIT", command = self.quit)
        # self.QUIT.pack(side="top")

        self.mymaster.ImageTitle = '../Images/image1.png'
        self.img = ImageTk.PhotoImage(Image.open(self.mymaster.ImageTitle))
        self.mymaster.InputImagePanel.config(image = self.img)
        self.mymaster.update()

def main():
    root = Tk()

    root.ImageTitle = ""
    root.InputImagePanel = Label()
    root.InputImagePanel.pack(side = "left", fill = "both", expand = "yes")
    root.OutputImagePanel = Label()
    root.OutputImagePanel.pack(side = "right", fill = "both", expand = "yes")
    root.wm_title("draw the Graph")
    # root.configure(background="black")

    root.geometry("1000x800")    # size
    root.resizable(width=FALSE, height=FALSE)   # fixed size
    myDropDownMenu = dropDownMenu(root)

    app = Application(root)

    app.mainloop()

    root.destroy()

if __name__ == '__main__':
    main()