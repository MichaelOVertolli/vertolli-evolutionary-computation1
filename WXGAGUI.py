###############################################################################
#Copyright (C) 2013  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

"""
This is a hack job of a GUI setup that was designed to be convenient for showing
each of the GA problems quickly. It is not thoroughly implemented as the GUI was
not a requirement of the project nor was the threading setup.

Classes:
GAThread(size, tSize, mProb, cProb, tProb, rProb, elite, k,
         dir_, gens, adaptiveMod, file_, question, selFunction,
         mutFunction, crossFunction, query)
         -- modified thread class for the underlying GA processing
WXGAGUI(parent, title) -- base frame for GUI
UI(parent) -- panel for interface
TSPScreen(parent) -- scrolling text window for display text
TSPCanvas(parent) -- panel for drawing the TSP graphic on a canvas
MyApp() -- main function that runs the entire program
"""

import operator
import os
import string
import wx
import wx.lib.agw.floatspin as FS
from AbstractGA import AbstractGA
from functools import partial
from Question1 import StrSearch
from Question2 import OneMax, SimpleMax, LeadingOnes
from Question3 import TSPGA
from threading import Thread
from wx.lib.pubsub import Publisher


class GAThread(Thread):
  """Modified thread class for the underlying GA processing.

  Uses Publisher to communicate between GUI and itself.

  Public Methods:
  None

  """
  def __init__(self, size, tSize, mProb, cProb, tProb, rProb, elite, k,
               dir_, gens, adaptiveMod, file_, question, selFunction,
               mutFunction, crossFunction, query):
    """Sets up all the relevant properties for a particular GA problem.

    Keyword arguments:
    size (int) -- the size of the population
    tSize (int) -- size of the tournaments
    mProb (float) -- probability of mutation
    cProb (float) -- probability of crossover
    tProb (float) -- probability of random tournament selection
    rProb (float) -- selection pressure for rank selection
    elite (bool) -- indicates if keeping best chromosome overall (elitism)
    k (int) -- pool size for heuristic mutation
    dir_ (int) -- max (-1) or min (0) based selection
    gens (int) -- max number of generations for one problem run
    adaptiveMod (bool) -- use adaptive modification of probabilities
    file_ (string) -- travelling salesman representation file
    question (string) -- string indicating which GA problem
    selFunction (func) -- a selection function
    mutFunction (func) -- a mutation function
    crossFunction (func) -- a crossover function
    query (string) -- query string for StringSearch and certain tests
    
    """
    self.question = question
    self.mapping =  {'Roulette' : 'rouletteSel',
                     'Linear Rank' : 'rankSel',
                     'Tournament' : 'tournamentSel',
                     'Probability' : 'basicMutate',
                     'Inversion' : 'inversionMutate',
                     'Insertion' : 'insertionMutate',
                     'Reciprocal' : 'reciprocalMutate',
                     'Heuristic' : 'heuristicMutate',
                     '1-Point' : 'oneCrossover', 
                     '2-Point' : 'twoCrossover', 
                     'Partial Map' : 'partialCrossover', 
                     'Order' : 'orderCrossover', 
                     'Injection' : 'injectionCrossover', 
                     'Position' : 'positionCrossover', 
                     'Recombination' : 'edgeRecombUI',
                     'StringSearch' : 'getClosest',
                     'True' : True,
                     'False' : False}
    self.divider=5
    adaptiveMod = self.mapping[adaptiveMod]
    elite = self.mapping[elite]

    #This switches the direction of the fitness functions.
    #-1 takes the right-most in the list, which is the highest value.
    #0 takes the left-most in the list, which is the lowest value.
    if dir_ == 'Max':
      dir_ = -1
    else:
      dir_ = 0
    if self.question == 'TSP':
      if file_ == None:
        error = wx.MessageDialog(None, 'You must select a file.',
                               'File Error', wx.OK|wx.ICON_ERROR)
        error.ShowModal()
        Publisher().sendMessage('Done', True)
        return
      self.solver = TSPGA(size, tSize, mProb, cProb, tProb, rProb, elite,
                          k, dir_, gens, file_)
    elif self.question == 'OneMax':
      self.solver = OneMax(size, tSize, mProb, cProb, tProb, rProb, elite,
                          k, dir_, gens, min_=0, max_=1, noise=1, gSize=30)
    elif self.question == 'SimpleMax':
      self.solver = SimpleMax(size, tSize, mProb, cProb, tProb, rProb, elite,
                          k, dir_, gens, min_=1, max_=10, noise=5, gSize=10)
    elif self.question == 'LeadingOnes':
      self.solver = LeadingOnes(size, tSize, mProb, cProb, tProb, rProb, elite,
                          k, dir_, gens, min_=0, max_=1, noise=1, gSize=30)
    elif self.question == 'StringSearch':
      if query == '':
        error = wx.MessageDialog(None, 'You must enter a query.',
                               'Text Error', wx.OK|wx.ICON_ERROR)
        error.ShowModal()
        Publisher().sendMessage('Done', True)
        return
      self.solver = StrSearch(size, tSize, mProb, cProb, tProb, rProb, elite,
                              k, dir_, gens, min_=0, max_=26, noise=5,
                              gSize=len(query), query=query)
      selFunction = self.question
    elif self.question == 'Test Crossover':
      self.solver = AbstractGA()
      fn = getattr(self.solver, self.mapping[crossFunction])
      result = fn(query[0], query[1])
      Publisher().sendMessage('Print', result)
      Publisher().sendMessage('Done', True)
      return
    else:
      error = wx.MessageDialog(None, 'You must select a question.',
                               'Question Error', wx.OK|wx.ICON_ERROR)
      error.ShowModal()
      Publisher().sendMessage('Done', True)
      return
    self.selFunction = getattr(self.solver, self.mapping[selFunction])
    self.mutFunction = getattr(self.solver, self.mapping[mutFunction])
    self.crossFunction = getattr(self.solver, self.mapping[crossFunction])
    if adaptiveMod:
      if dir_ == -1:
        self.process = partial(self.solver.run, self.selFunction, self.mutFunction,
                           self.crossFunction, operator.gt)
      else:
        self.process = partial(self.solver.run, self.selFunction, self.mutFunction,
                           self.crossFunction, operator.lt)
    else:
      self.process = partial(self.solver.run, self.selFunction, self.mutFunction,
                         self.crossFunction)
    Thread.__init__(self)
    self.running = True
    Publisher().subscribe(self.die, 'Death')
    self.start()

  def run(self):
    """Base thread function for GA logic processing."""
    while self.solver.gens < self.solver.MAX_GENS and self.running:
      results = self.process()
      try:
        Publisher().sendMessage('Draw', self.solver.prep(results[0],
                                                         self.divider))
      except AttributeError:
        pass
      Publisher().sendMessage('Print', (results[0],
                                        self.solver.fitFunction(results[0]),
                                        results[1]))
    Publisher().sendMessage('Print',
                            ''.join(['....Finished....\n\n',
                                'Best: '+str(results[0])+'\n\n',
                                'Value: '+str(
                                    self.solver.fitFunction(results[0]))+'\n\n',
                                'Generations: '+str(results[1])+'\n\n']))
    try:
      Publisher().sendMessage('Draw', self.solver.prep(results[0], self.divider))
    except AttributeError:
      pass
    Publisher().sendMessage('Done', True)

  def die(self, msg):
    """Sets an internal variable false to kill the thread."""
    self.running = False

class WXGAGUI(wx.Frame):
  """Modified frame class for the base GUI structure.

  Public Methods:
  None

  """
  def __init__(self, parent, title):
    """Sets up all the base panels for the GUI.

    Keyword arguments:
    parent (None) -- should be set to None when called
    title (string) -- name to appear in window title bar

    """
    super(WXGAGUI, self).__init__(parent, wx.ID_ANY, title, size=(1200, 700))

    self.InitUI()
    self.Centre()

  def InitUI(self):
    panel = wx.Panel(self)
    panel.SetBackgroundColour('Black')
    panelB = UI(panel)
    panelT = wx.Panel(panel, size=(1000, 600))
    panelT.SetBackgroundColour('Red')
    panelL = TSPScreen(panelT)
    panelR = TSPCanvas(panelT)
    vbox = wx.BoxSizer(wx.VERTICAL)
    vbox.Add(panelT, 2, wx.ALIGN_TOP|wx.EXPAND)
    vbox.Add(panelB, 1, wx.ALIGN_BOTTOM|wx.EXPAND)
    hbox = wx.BoxSizer(wx.HORIZONTAL)
    hbox.Add(panelL, 1, wx.ALIGN_LEFT|wx.EXPAND)
    hbox.Add(panelR, 1, wx.ALIGN_RIGHT|wx.EXPAND)
    
    panel.SetSizer(vbox)
    panelT.SetSizer(hbox)

    self.Layout()

class UI(wx.Panel):
  """Modified panel class for the UI.

  Public Methods:
  None

  """
  def __init__(self, parent):
    """Sets up all the panels for the selection interface.

    Keyword arguments:
    parent (panel) -- panel that this panel is attached to

    """
    super(UI, self).__init__(parent, wx.ID_ANY, size=(1200, 100),
                                    style=wx.RAISED_BORDER)
    self.SetBackgroundColour('Light Grey')

    #For opening TSP representation file
    self.cd = os.getcwd()
    self.invalid = set(string.printable) - set(string.lowercase) - set([' '])
    self.query = ''

    hbox = wx.BoxSizer(wx.HORIZONTAL)
    fgr = wx.FlexGridSizer(3, 6, 10, 10)

    #For base interface text
    questionL = wx.StaticText(self, label='Question:')
    selTypeL = wx.StaticText(self, label='Selection Type:')
    mutTypeL = wx.StaticText(self, label='Mutation Type:')
    crossTypeL = wx.StaticText(self, label='Crossover Type:')
    popL = wx.StaticText(self, label='Population:')
    mutL = wx.StaticText(self, label='Mutation:')
    crossL = wx.StaticText(self, label='Crossover:')
    genL = wx.StaticText(self, label='Generations:')
    eliteL = wx.StaticText(self, label='Elite:')
    tSizeL = wx.StaticText(self, label='Tourney Size:')
    tProbL = wx.StaticText(self, label='Tourney Prob:')
    rProbL = wx.StaticText(self, label='Rank Pressure:')
    hKL = wx.StaticText(self, label='Heuristic K:')
    dirL = wx.StaticText(self, label='Max/Min:')
    adaptiveModL = wx.StaticText(self, label='Adaptive Prob:')

    #For selection widgets    
    self.question = wx.ComboBox(self, choices=['StringSearch', 'OneMax',
                                               'SimpleMax', 'LeadingOnes',
                                               'TSP', 'Test Crossover'],
                                style=wx.CB_READONLY)
    self.sel = wx.ComboBox(self, choices=['Roulette', 'Linear Rank',
                                          'Tournament'], style=wx.CB_READONLY)
    self.mut = wx.ComboBox(self, choices=['Probability', 'Insertion',
                                          'Inversion', 'Reciprocal',
                                          'Heuristic'], style=wx.CB_READONLY)
    self.cross = wx.ComboBox(self, choices=['1-Point', '2-Point', 'Partial Map',
                                            'Order', 'Injection', 'Position',
                                            'Recombination'],
                             style=wx.CB_READONLY)
    self.elite = wx.ComboBox(self, choices=['True', 'False'],
                             style=wx.CB_READONLY)
    self.dir_ = wx.ComboBox(self, choices=['Max', 'Min'], style=wx.CB_READONLY)
    self.adaptiveMod = wx.ComboBox(self, choices=['True', 'False'],
                                   style=wx.CB_READONLY)

    self.popSize = FS.FloatSpin(self, min_val=500, max_val=2000, value=500,
                                increment=100, digits=0)
    self.mProb = FS.FloatSpin(self, min_val=0.001, max_val=0.5, value=0.001,
                              increment=0.001, digits=3)
    self.cProb = FS.FloatSpin(self, min_val=0.01, max_val=0.99, value=0.01,
                              increment=0.01, digits=2)
    self.gensMax = FS.FloatSpin(self, min_val=30, max_val=2000, value=30,
                                increment=10, digits=0)
    self.tSize = FS.FloatSpin(self, min_val=5, max_val=500, value=5,
                              increment=5, digits=0)
    self.tProb = FS.FloatSpin(self, min_val=0.1, max_val=1.0, value=0.1,
                              increment=0.1, digits=1)
    self.rProb = FS.FloatSpin(self, min_val=1.0, max_val=2.0, value=1.0,
                           increment=0.1, digits=1)
    self.k = FS.FloatSpin(self, min_val=2, max_val=30, value=2, increment=1,
                          digits=0)

    #For action buttons and event bindings
    self.openBtn = wx.Button(self, label='OPEN', size=(100, 50))
    self.openBtn.Bind(wx.EVT_BUTTON, self.Open)
    self.runBtn = wx.Button(self, label='RUN', size=(100, 50))
    self.runBtn.Bind(wx.EVT_BUTTON, self.Run)
    self.stopBtn = wx.Button(self, label='STOP', size=(100, 50))
    self.stopBtn.Bind(wx.EVT_BUTTON, self.Kill)

    self.txtEntr = wx.TextCtrl(self, size=(590, 20),
                               style=wx.TE_PROCESS_ENTER)
    self.txtEntr.Bind(wx.EVT_TEXT_ENTER, self.Enter)

    #Placing all the elements
    hbox2 = wx.BoxSizer(wx.HORIZONTAL)
    hbox2.Add(self.txtEntr, 1, wx.EXPAND)
    vbox = wx.BoxSizer(wx.VERTICAL)

    fgr.AddMany([questionL, self.question, selTypeL, self.sel,
                 mutTypeL, self.mut, crossTypeL, self.cross,
                 eliteL, self.elite, popL, self.popSize, mutL,
                 self.mProb, crossL, self.cProb, genL, self.gensMax,
                 tSizeL, self.tSize, tProbL, self.tProb, rProbL, self.rProb,
                 hKL, self.k, dirL, self.dir_, adaptiveModL, self.adaptiveMod])

    vbox.Add(hbox2, 0, wx.ALIGN_LEFT)
    vbox.Add(fgr, 1, wx.ALIGN_LEFT|wx.ALIGN_BOTTOM|wx.TOP|wx.LEFT, 5)
    hbox.Add(vbox, 1, wx.ALIGN_LEFT)
    hbox.Add(self.openBtn, 0, wx.ALIGN_RIGHT|wx.ALIGN_BOTTOM)
    hbox.Add(self.stopBtn, 0, wx.ALIGN_RIGHT|wx.ALIGN_BOTTOM)
    hbox.Add(self.runBtn, 0, wx.ALIGN_RIGHT|wx.ALIGN_BOTTOM)
    self.SetSizer(hbox)
    self.Layout()

    self.question.Bind(wx.EVT_COMBOBOX, self.QSelect)

    Publisher().subscribe(self.Reset, 'Done')

  def Enter(self, event):
    """Evaluates when Enter key is pressed in text enter box."""
    txt = event.GetClientObject()
    a = txt.GetValue()
    q = self.question.GetValue()
    if q == 'StringSearch':
      if len(set(a) & self.invalid) > 0:
        error = wx.MessageDialog(None, 'Only lowercase letters are valid.',
                                 'Text Error', wx.OK|wx.ICON_ERROR)
        error.ShowModal()
      elif len(a) < 30:
        error = wx.MessageDialog(None, 'You must enter at least 30 characters.',
                                 'Text Error', wx.OK|wx.ICON_ERROR)
        error.ShowModal()
      else:
        self.query = a
        Publisher().sendMessage('Print', a)
        #txt.SetStyle(wx.TE_READONLY)
    elif q == 'Test Crossover':
      a = a.replace(' ', '')
      b = a.split(',')
      self.query = [list(b[0]), list(b[1])]
      Publisher().sendMessage('Print', self.query)
    txt.Clear()

  def QSelect(self, event):
    """Selects default settings for each GA Problem."""
    q = event.GetString()
    if q == 'TSP':
      self.popSize.SetValue(500)
      self.tSize.SetValue(10)
      self.mProb.SetValue(0.001)
      self.cProb.SetValue(0.7)
      self.tProb.SetValue(0.6)
      self.rProb.SetValue(1.2)
      self.elite.SetValue('False')
      self.k.SetValue(2)
      self.dir_.SetValue('Min')
      self.adaptiveMod.SetValue('True')
      self.gensMax.SetValue(2000)
      self.file = 'berlin52.txt'
      self.sel.SetValue('Tournament')
      self.mut.SetValue('Inversion')
      self.cross.SetValue('Recombination')
    elif q == 'OneMax':
      self.popSize.SetValue(1000)
      self.tSize.SetValue(10)
      self.mProb.SetValue(0.001)
      self.cProb.SetValue(0.8)
      self.tProb.SetValue(0.6)
      self.rProb.SetValue(1.2)
      self.elite.SetValue('True')
      self.k.SetValue(2)
      self.dir_.SetValue('Max')
      self.adaptiveMod.SetValue('False')
      self.gensMax.SetValue(50)
      self.file = None
      self.sel.SetValue('Roulette')
      self.mut.SetValue('Probability')
      self.cross.SetValue('1-Point')
    elif q == 'SimpleMax':
      self.popSize.SetValue(1000)
      self.tSize.SetValue(10)
      self.mProb.SetValue(0.001)
      self.cProb.SetValue(0.8)
      self.tProb.SetValue(0.6)
      self.rProb.SetValue(1.2)
      self.elite.SetValue('True')
      self.k.SetValue(2)
      self.dir_.SetValue('Max')
      self.adaptiveMod.SetValue('False')
      self.gensMax.SetValue(20)
      self.file = None
      self.sel.SetValue('Roulette')
      self.mut.SetValue('Probability')
      self.cross.SetValue('1-Point')
    elif q == 'LeadingOnes':
      self.popSize.SetValue(1000)
      self.tSize.SetValue(10)
      self.mProb.SetValue(0.001)
      self.cProb.SetValue(0.8)
      self.tProb.SetValue(0.6)
      self.rProb.SetValue(1.2)
      self.elite.SetValue('True')
      self.k.SetValue(2)
      self.dir_.SetValue('Max')
      self.adaptiveMod.SetValue('False')
      self.gensMax.SetValue(50)
      self.file = None
      self.sel.SetValue('Roulette')
      self.mut.SetValue('Probability')
      self.cross.SetValue('1-Point')
    elif q == 'StringSearch':
      self.popSize.SetValue(500)
      self.tSize.SetValue(10)
      self.mProb.SetValue(0.1)
      self.cProb.SetValue(0.8)
      self.tProb.SetValue(0.6)
      self.rProb.SetValue(1.2)
      self.elite.SetValue('True')
      self.k.SetValue(2)
      self.dir_.SetValue('Min')
      self.adaptiveMod.SetValue('True')
      self.gensMax.SetValue(200)
      self.file = None
      self.sel.SetValue('Roulette')
      self.mut.SetValue('Probability')
      self.cross.SetValue('1-Point')
      Publisher().sendMessage('Print', """Enter a string of at least 30 \
lowercase letters with no punctuation.""")
    elif q == 'Test Crossover':
      self.popSize.SetValue(1000)
      self.tSize.SetValue(10)
      self.mProb.SetValue(0.001)
      self.cProb.SetValue(0.99)
      self.tProb.SetValue(0.6)
      self.rProb.SetValue(1.2)
      self.elite.SetValue('False')
      self.k.SetValue(2)
      self.dir_.SetValue('Max')
      self.adaptiveMod.SetValue('False')
      self.gensMax.SetValue(50)
      self.file = None
      self.sel.SetValue('Roulette')
      self.mut.SetValue('Probability')
      self.cross.SetValue('2-Point')
      Publisher().sendMessage('Print', """Please enter two strings separated \
by a comma.""")

  def Run(self, event):
    """Sends all the selected values to the GA logic processing thread.

    Disables run button after start.

    """
    btn = event.GetEventObject()
    btn.Disable()
    self.thread = GAThread(int(self.popSize.GetValue()),
                           int(self.tSize.GetValue()), self.mProb.GetValue(),
                           self.cProb.GetValue(), self.tProb.GetValue(),
                           self.rProb.GetValue(), self.elite.GetValue(),
                           int(self.k.GetValue()), self.dir_.GetValue(),
                           int(self.gensMax.GetValue()),
                           self.adaptiveMod.GetValue(), self.file,
                           self.question.GetValue(), self.sel.GetValue(),
                           self.mut.GetValue(), self.cross.GetValue(),
                           self.query)

  def Reset(self, event):
    """Enables run button after problem complete or stopped."""
    self.runBtn.Enable()

  def Kill(self, event):
    """Ends GA process on GAThread before completion."""
    Publisher().sendMessage('Death', True)

  def Open(self, event):
    """Sets up open dialogue box for selecting TSP representation file."""
    dlg = wx.FileDialog(self, message='Choose a file', defaultDir=self.cd,
                        defaultFile='',
                        style=wx.OPEN|wx.CHANGE_DIR)
    if dlg.ShowModal() == wx.ID_OK:
      self.file = dlg.GetPath()

class TSPScreen(wx.ScrolledWindow):
  """Modified scrolling window class for the text display.

  Public Methods:
  None

  """
  def __init__(self, parent):
    """Sets up the scrolling window for the text display.

    Keyword arguments:
    parent (panel) -- panel that this window is attached to

    """
    super(TSPScreen, self).__init__(parent, wx.ID_ANY, size=(600, 600),
                                    style=wx.SUNKEN_BORDER)

    self.SetBackgroundColour('White')
    self.SetVirtualSize((500, 600))
    self.SetScrollRate(0, 10)

    vbox = wx.BoxSizer(wx.VERTICAL)

    self.text = wx.TextCtrl(self,
                            value='Started...\nPlease select question type.\n',
                            style=wx.TE_MULTILINE|wx.TE_READONLY)
    self.text.SetInsertionPointEnd()

    vbox.Add(self.text, 1, wx.EXPAND)
    self.SetSizer(vbox)

    self.Layout()

    Publisher().subscribe(self.PrintTSP, 'Print')

  def PrintTSP(self, msg):
    """Function for printing text to this window."""
    msg = str(msg.data)
    self.text.WriteText(msg+'\n')

class TSPCanvas(wx.Panel):
  """Modified panel for drawing the TSP graphic on a canvas.

  Public Methods:
  None

  """
  def __init__(self, parent):
    """Sets up the canvas for drawing the TSP current solution.

    Keyword arguments:
    parent (panel) -- panel that this panel is attached to

    """
    super(TSPCanvas, self).__init__(parent, wx.ID_ANY, size=(600, 600),
                                    style=wx.SUNKEN_BORDER)

    self.SetBackgroundColour('White')
    self.color = 'Black'
    self.thickness = 1
    self.pen = wx.Pen(self.color, self.thickness, wx.SOLID)
    self.lines = []
    self.InitBuffer()

    self.Bind(wx.EVT_SIZE, self.OnSize)
    self.Bind(wx.EVT_IDLE, self.OnIdle)
    self.Bind(wx.EVT_PAINT, self.OnPaint)

    self.Layout()
    
    Publisher().subscribe(self.DrawTSP, 'Draw')

  def InitBuffer(self):
    """Drawing buffer that is blitted to the screen."""
    size = self.GetClientSize()
    self.buffer = wx.EmptyBitmap(size.width, size.height)
    dc = wx.BufferedDC(None, self.buffer)
    dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
    dc.Clear()
    self.DrawLines(dc)
    self.reInitBuffer = False

  def GetLinesData(self):
    """Convenience function that returns a copy of lines to be drawn."""
    return self.lines[:]

  def SetLinesData(self, lines):
    """Sets the current line data with lines, setup buffer and blit.

    Keyword arguments:
    lines (list) -- a list of [x1, y1, x2, y2] lists

    """
    self.lines = lines[:]
    self.InitBuffer()
    self.Refresh()

  def OnSize(self, event):
    """Redraws the screen if the screen is resized."""
    self.reInitBuffer = True

  def OnIdle(self, event):
    """Draws to the screen if nothing is happening."""
    if self.reInitBuffer:
      self.InitBuffer()
      self.Refresh(False)

  def OnPaint(self, event):
    """Drawing function."""
    dc = wx.BufferedPaintDC(self, self.buffer)

  def DrawLines(self, dc):
    """Drawing process."""
    if self.lines:
      for coords in self.lines:
        dc.DrawLine(*coords)

  def DrawTSP(self, msg):
    """The initial function that is called to initial the draw process.

    Keyword arguments:
    msg (obj) -- Publisher object with message passing data

    """
    lines = [x for x in msg.data]
    self.SetLinesData(lines)

class MyApp(wx.App):
  """Modified app for running the base GUI thread.

  Public Methods:
  MainLoop() -- starts the process

  """
  def OnInit(self):
    """Setup function."""
    frame = WXGAGUI(None, 'GA Program')
    frame.Show(True)
    return True

if __name__ == '__main__':
  app = MyApp(0)
  app.MainLoop()

    
