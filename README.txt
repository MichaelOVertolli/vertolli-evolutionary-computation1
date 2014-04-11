Thank you for being interested in my work!

This is a rather extensive project I made for a course on evolutionary programming.
In order to make my work easier, I created various abstractions of the basic setup
and then stitched them together to solve a particular problem for the class (see,
in particular, Question2.py where entire problems are implemented in 4 lines of 
code). I added a simple (if not robust) GUI to making moving across problems simpler.

AbstractGA describes the basic functionality for the GA class
GenericGA sets up the basic properties and structure for a GA algorithm
Question1 through 3 are implementations of particular problems that can be solved with
a GA
WXGAGUI is the GUI and threading framework that stitches it all together.

The required Python libraries for this to run are:
wxPython (http://www.wxpython.org/download.php)

With the appropriate libraries installed, you can just double-click on the WXGAGUI.py
file to run the program.

If you do not have or do not want to install them, the dist.zip file contains an
executable version of the python code built with py2exe.

In order to run the program you might need to download MSVCR90.dll, which you can
find here: http://www.microsoft.com/en-us/download/details.aspx?id=29

The explanation is here: http://www.py2exe.org/index.cgi/Tutorial

Once unzipped, one need only go to the dist/WXGAGUI.exe to run.

INSTRUCTIONS FOR GUI

The workflow for the GUI is as follows:
Select a question from the Question drop down menu.
Default settings will be set for the different functions and variables.
Many of the various features can be found explained online, including different
mutation, selection, and crossover operators. Details are also included in the
GenericGA.py file.
Once the setup is okay, click Run.
Stop will terminate the given problem.
Open is used only for the TSP problem if you want to select a harder TSP
representation, mainly holycross.tsp.
The Evo Strategy problem also requires a minimum 30 character input string in all 
lowercase letters.
Test Crossover requires input of the form: 111111, 000000
with whatever characters you want. The u's indicate that the characters are in 
unicode.
