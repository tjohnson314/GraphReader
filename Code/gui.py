from Tkinter import *
from tkFileDialog import askopenfilename #
from PIL import ImageTk, Image

class dropDownMenu:
    label3 = None
    def __init__(self,master):
        self.mymaster = master
        self.filename=""
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
        self.filename = askopenfilename()
        self.img = ImageTk.PhotoImage(Image.open(self.filename))
        self.mymaster.InputImagePanel.config(image = self.img)
        self.mymaster.update()


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.mymaster = master
        self.pack()
        self.createWidgets()

    def process(self):
        print "processing here!"

    def createWidgets(self):

        self.process = Button(self, text = "process", command = self.process)
        self.process.pack(side="top",fill = "both", expand = "yes", padx=15, pady=15)

        # self.QUIT = Button(self, text = "QUIT", command = self.quit)
        # self.QUIT.pack(side="top")

        self.path = 'hacktech.png'
        self.img = ImageTk.PhotoImage(Image.open(self.path))
        self.mymaster.InputImagePanel.config(image = self.img)
        self.mymaster.update()

def main():
    root = Tk()

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

main()