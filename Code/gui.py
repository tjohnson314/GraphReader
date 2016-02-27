from Tkinter import *
from tkFileDialog import askopenfilename #
import PIL
from PIL import ImageTk
from PIL import Image, ImageDraw

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
        self.tempImage = Image.open(self.mymaster.ImageTitle)
        self.tempImage = self.tempImage.resize((470, 520), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage( self.tempImage)
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

        width = 400
        height = 400
        white = (255, 255, 255)
        black = (0,0,0)
        radius = 3;

        imageOutput = Image.new("RGB", (width, height), white)
        draw = ImageDraw.Draw(imageOutput)

        for node in myGraph.nodes():
            xPos = node.x
            yPos = node.y
            # Draw node at (xPos, yPos)
            draw.circle((xPos+radius,yPos+radius,xPos-radius,yPos-radius), black)

        for edge in myGraph.edges():
            x0 = edge[0].x
            y0 = edge[0].y
            x1 = edge[1].x
            y1 = edge[1].y
            # Draw line from (x0, y0) to (x1, y1)
            draw.line([(x0,y0),(x1,y1)],black)

        outputTitle = self.appendAfterLastSlash(self.mymaster.ImageTitle, "Output")
        imageOutput.save(outputTitle)

        self.img = ImageTk.PhotoImage(Image.open(outputTitle))
        self.mymaster.OutputImagePanel.config(image = self.img)
        self.mymaster.update()

    def createWidgets(self):

        self.process = Button(self, text = "process", command = self.process)
        self.process.pack(side="top",fill = "both", expand = "yes", padx=15, pady=15)

        # self.QUIT = Button(self, text = "QUIT", command = self.quit)
        # self.QUIT.pack(side="top")

        self.mymaster.ImageTitle = 'hacktech.png'
        self.tempImage = Image.open(self.mymaster.ImageTitle)
        self.tempImage = self.tempImage.resize((470, 520), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage( self.tempImage)
        self.mymaster.InputImagePanel.config(image = self.img)
        self.mymaster.update()


    def appendAfterLastSlash(self, s, toAppend):
        output = "";
        foundSlash = False
        for i in range(len(s) - 1, -1, -1):
            if not foundSlash and s[i] == '/':
                output = '/' + toAppend + output
                foundSlash = True
            else:
                output = s[i] + output

        return output

def make_panel(master, x, y, h, w, *args, **kwargs):
    f = Frame(master, height=h, width=w)
    f.pack_propagate(0) # don't shrink
    f.place(x=x, y=y)
    label = Label(f, *args, **kwargs)
    label.pack(fill=BOTH, expand=1)
    return label


def main():
    root = Tk()

    root.ImageTitle = ""
    root.InputImagePanel = make_panel(root, 10, 60, 580, 480, text='Input', background='white')
    root.OutputImagePanel = make_panel(root, 510, 60, 580, 480, text='Output', background='white')

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