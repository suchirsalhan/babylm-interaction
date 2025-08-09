#Summary Analysis Research Tool
#Takes a source text and summaries of source text as input
#Returns a number of indices for summary texts as output

from __future__ import division
import Tkinter as tk
import tkFont
import tkFileDialog
import Tkconstants
import os
import re
import sys 
import platform
import shutil
import subprocess
import glob
from threading import Thread
import Queue
try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET

import ktatk as ktk #this is Kris' toolkit

def resource_path(relative):
	if hasattr(sys, "_MEIPASS"):
		return os.path.join(sys._MEIPASS, relative)
	return os.path.join(relative)

#This creates a que in which the core program can communicate with the GUI
dataQueue = Queue.Queue()

#This creates the message for the progress box (and puts it in the dataQueue)
progress = "...Waiting for Data to Process"
dataQueue.put(progress)

#Def1 is the core program; args is information from GUI passed to program
def start_thread(def1, arg1, arg2, arg3,arg4): 
	t = Thread(target=def1, args=(arg1, arg2, arg3, arg4))
	t.start()

if platform.system() == "Darwin":
	system = "M"
	title_size = 16
	font_size = 14
	geom_size = "425x400"
	color = "#E6E6E6"
elif platform.system() == "Windows":
	system = "W"
	title_size = 14
	font_size = 12
	geom_size = "425x400"
	color = "#E6E6E6"
elif platform.system() == "Linux":
	system = "L"
	title_size = 14
	font_size = 12
	geom_size = "425x400"
	color = "#E6E6E6"

class MyApp: #this is the class for the gui and the text analysis
	def __init__(self, parent):
		
		#Creates font styles: Consider changing to Lucida Grande or Helvetica Neue
		helv14= tkFont.Font(family= "Helvetica Neue", size=font_size)
		times14= tkFont.Font(family= "Lucida Grande", size=font_size)
		helv16= tkFont.Font(family= "Helvetica Neue", size = title_size, weight = "bold", slant = "italic")
				#This defines the GUI parent (ish)
		
		self.myParent = parent
		
		#This creates the header text - Task:work with this to make more pretty!
		self.spacer1= tk.Label(parent, text= "Constructed Response Analysis Tool", font = helv16, background = color)
		self.spacer1.pack()
		
		#This creates a frame for the meat of the GUI
		self.thestuff= tk.Frame(parent, background =color)
		self.thestuff.pack()
		
		self.myContainer1= tk.Frame(self.thestuff, background = color)
		self.myContainer1.pack(side = tk.RIGHT, expand= tk.TRUE)

		self.labelframe2 = tk.LabelFrame(self.myContainer1, text= "Instructions", background = color)
		self.labelframe2.pack(expand=tk.TRUE)
		
		#This creates the list of instructions.
		self.instruct = tk.Button(self.myContainer1, text = "Instructions", justify = tk.LEFT)
		self.instruct.pack()
		self.instruct.bind("<Button-1>", self.instruct_mess)

		self.var_list = [] #passes to main program. can be used with checkboxes for options

		self.secondframe= tk.LabelFrame(self.myContainer1, text= "Data Input", background = color)
		self.secondframe.pack(expand=tk.TRUE) 

		self.summary_name = ""
		self.summary_button = tk.Button(self.secondframe)
		self.summary_button.configure(text= "Select Summary Source Text")
		self.summary_button.pack()
		self.summary_button.bind("<Button-1>", self.get_summary_text)

		self.summarylabel =tk.LabelFrame(self.secondframe, height = "1", width= "45", padx = "4", text = "Your selected summary text:", background = color)
		self.summarylabel.pack()

		summary_file_name = "(No Summary Text Chosen)"
		self.summarylabelchosen = tk.Label(self.summarylabel, height= "1", width= "44", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text = summary_file_name)
		self.summarylabelchosen.pack()

		#This Places the first button under the instructions.
		self.button1 = tk.Button(self.secondframe)
		self.button1.configure(text= "Select Input Folder")
		self.button1.pack()
		
		#This tells the button what to do when clicked.	 Currently, only a left-click
		#makes the button do anything (e.g. <Button-1>). The second argument is a "def"
		#That is defined later in the program.
		self.button1.bind("<Button-1>", self.button1Click)
		
		#Creates default dirname so if statement in Process Texts can check to see
		#if a directory name has been chosen
		self.dirname = ""
		
		#This creates a label for the first program input (Input Directory)
		self.inputdirlabel =tk.LabelFrame(self.secondframe, height = "1", width= "45", padx = "4", text = "Your selected input folder:", background = color)
		self.inputdirlabel.pack()
		
		#Creates label that informs user which directory has been chosen
		directoryprompt = "(No Folder Chosen)"
		self.inputdirchosen = tk.Label(self.inputdirlabel, height= "1", width= "44", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text = directoryprompt)
		self.inputdirchosen.pack()
		
		#This creates the Output Directory button.
		self.button2 = tk.Button(self.secondframe)
		self.button2["text"]= "Select Output Filename"
		#This tells the button what to do if clicked.
		self.button2.bind("<Button-1>", self.button2Click)
		self.button2.pack()
		self.outdirname = ""
		
		#Creates a label for the second program input (Output Directory)
		self.outputdirlabel = tk.LabelFrame(self.secondframe, height = "1", width= "45", padx = "4", text = "Your selected output filename:", background = color)
		self.outputdirlabel.pack()
		
		#Creates a label that informs sure which directory has been chosen
		outdirectoryprompt = "(No Output Filename Chosen)"
		self.outputdirchosen = tk.Label(self.outputdirlabel, height= "1", width= "44", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text = outdirectoryprompt)
		self.outputdirchosen.pack()

		self.BottomSpace= tk.LabelFrame(self.myContainer1, text = "Run Program", background = color)
		self.BottomSpace.pack()

		self.button3= tk.Button(self.BottomSpace)
		self.button3["text"] = "Process Texts"
		self.button3.bind("<Button-1>", self.runprogram)
		self.button3.pack()

		self.progresslabelframe = tk.LabelFrame(self.BottomSpace, text= "Program Status", background = color)
		self.progresslabelframe.pack(expand= tk.TRUE)
		
		self.progress= tk.Label(self.progresslabelframe, height= "1", width= "45", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text=progress)
		self.progress.pack()
		
		self.poll(self.progress)
	
	#### Change this in final Program!!!! #####
		#self.summary_name = "/Users/kriskyle/Desktop/stoic 1 files/stoicModule1Text.txt"
		#self.dirname = "/Users/kriskyle/Desktop/stoic 1 files/source files"
		#self.outdirname = "/Users/kriskyle/Desktop/results.csv"
	#### Change this in final Program!!!! #####

	
	def instruct_mess(self, event):
		import tkMessageBox
		tkMessageBox.showinfo("Instructions", "1. Select your summary source text.\n2. Choose the input folder (where your summaries are).\n3. Choose an output filename.\n4. Press the 'Process Texts' button.")

	def entry1Return(self,event):
		input= self.entry1.get()
		self.input2 = input + ".csv"
		self.filechosenchosen.config(text = self.input2)
		self.filechosenchosen.update_idletasks()

	def get_summary_text(self, event):
		import tkFileDialog
		self.summary_name = tkFileDialog.askopenfilename()
		self.displaysummary_file = '.../'+self.summary_name.split('/')[-1]
		self.summarylabelchosen.config(text = self.displaysummary_file)

		print self.summary_name
		
	def button1Click(self, event):
		#import Tkinter, 
		import tkFileDialog
		self.dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
		self.displayinputtext = '.../'+self.dirname.split('/')[-1]
		self.inputdirchosen.config(text = self.displayinputtext)
		
		#newmsg= "Chosen"
		#self.inputdirchosen.config(text = newmsg)
		#self.inputdirchosen.update_idletasks()

	def button2Click(self, event):
		#self.outdirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
		self.outdirname = tkFileDialog.asksaveasfilename(parent=root, defaultextension = ".csv", initialfile = "results",title='Choose Output Filename')
		print self.outdirname
		if self.outdirname == "":
			self.displayoutputtext = "(No Output Filename Chosen)"
		else: self.displayoutputtext = '.../' + self.outdirname.split('/')[-1]
		self.outputdirchosen.config(text = self.displayoutputtext)
		
	
	def SubmitFilenameButtonClick(self, event):
		input= self.entry1.get()
		self.input2 = input + ".csv"
		self.filechosenchosen.config(text = self.input2)
		
	def runprogram(self, event):
		self.poll(self.progress)
		start_thread(main, self.dirname, self.outdirname, self.summary_name, self.var_list)

	def poll(self, function):
		
		self.myParent.after(10, self.poll, function)
		try:
			function.config(text = dataQueue.get(block=False))
			
		except Queue.Empty:
			pass

def main(indir, outdir, summ_text, var_list):		
	import tkMessageBox
	if summ_text is "":
		tkMessageBox.showinfo("Supply Information", "Choose Source Summary Text")
	if indir is "":
		tkMessageBox.showinfo("Supply Information", "Choose Input Directory")
	if outdir is "":
		tkMessageBox.showinfo("Supply Information", "Choose Output Filename")
	if indir is not "" and outdir is not "":
	
		dataQueue.put("Starting SMART...")

		input_glob = indir + "/*.txt"
		
		outf=file(outdir, "w")
		
		key_out_dir = "/".join(outdir.split("/")[:-1])
		
		filenames = glob.glob(input_glob)
		
		for_stan = filenames
		for_stan.append(summ_text) #add summ_text for file processing

		
		file_number = 0
		file_counter = 1 #for update_list
		
		ktk.gui_stan_corenlp(system, ktk.call_stan_corenlp_pos, for_stan, "3", "2",dataQueue,root) #creates pos_tag version of the text

	#### Database Import and Dictionary Building ####
		dataQueue.put("Loading Program Databases")
		root.update_idletasks()

		dataQueue.put("Loading Psycholinguistic Norm Data...")

	#MRC Data
		mrc_list = file(resource_path('MRC_database_simple_final_lower.txt'), 'rU').read()
		fam = ktk.dict_builder(mrc_list, 6)
		concreteness = ktk.dict_builder(mrc_list, 7)
		imageability = ktk.dict_builder(mrc_list, 8)
		meaningfulness_colorado = ktk.dict_builder(mrc_list, 9)

	#Kuperman et al. AoA
		AoA_list= file(resource_path('AoA_Brysbart.txt'), 'rU').read()
		B_AoA = ktk.dict_builder(AoA_list, 1)
	
	#Brysbaert et al. Concreteness Ratings:
		B_conc_list= file(resource_path('Concreteness_Brysbaert.txt'), 'rU').read()
		B_Conc = ktk.dict_builder(B_conc_list, 1,)

		
	#SUBTLEXus:
		dataQueue.put("Loading SUBTLEXus Data...")
		freq_list = file(resource_path('SUBTLEXUS_lower.txt'), 'rU').read()
		subtlex_freq = ktk.dict_builder(freq_list, 1)
		subtlex_cd = ktk.dict_builder(freq_list, 2)
		subtlex_freq_log = ktk.dict_builder(freq_list, 4)
		subtlex_cd_log = ktk.dict_builder(freq_list, 6)

	#COCA Frequency and Range lists:
		dataQueue.put("Loading COCA Word Data... 1 of 4")
		root.update_idletasks()
		COCA_academic_uni_list = file(resource_path('COCA_academic_unigram_list.csv'), 'rU').read()
		COCA_academic_uni_R = ktk.dict_builder(COCA_academic_uni_list, 4)
		COCA_academic_uni_F = ktk.dict_builder(COCA_academic_uni_list, 2)
		COCA_academic_uni_R_log = ktk.dict_builder(COCA_academic_uni_list, 4,"y")
		COCA_academic_uni_F_log = ktk.dict_builder(COCA_academic_uni_list, 2,"y")

		dataQueue.put("Loading COCA Word Data... 2 of 4")
		COCA_fiction_uni_list = file(resource_path('COCA_fiction_unigram_list.csv'), 'rU').read()
		COCA_fiction_uni_R = ktk.dict_builder(COCA_fiction_uni_list, 4)
		COCA_fiction_uni_F = ktk.dict_builder(COCA_fiction_uni_list, 2)
		COCA_fiction_uni_R_log = ktk.dict_builder(COCA_fiction_uni_list, 4,"y")
		COCA_fiction_uni_F_log = ktk.dict_builder(COCA_fiction_uni_list, 2,"y")

		dataQueue.put("Loading COCA Word Data... 3 of 4")
		COCA_magazine_uni_list = file(resource_path('COCA_magazine_unigram_list.csv'), 'rU').read()
		COCA_magazine_uni_R = ktk.dict_builder(COCA_magazine_uni_list, 4)
		COCA_magazine_uni_F = ktk.dict_builder(COCA_magazine_uni_list, 2)
		COCA_magazine_uni_R_log = ktk.dict_builder(COCA_magazine_uni_list, 4,"y")
		COCA_magazine_uni_F_log = ktk.dict_builder(COCA_magazine_uni_list, 2,"y")

		dataQueue.put("Loading COCA Word Data... 4 of 4")
		COCA_news_uni_list = file(resource_path('COCA_newspaper_unigram_list.csv'), 'rU').read()
		COCA_news_uni_R = ktk.dict_builder(COCA_news_uni_list, 4)
		COCA_news_uni_F = ktk.dict_builder(COCA_news_uni_list, 2)
		COCA_news_uni_R_log = ktk.dict_builder(COCA_news_uni_list, 4,"y")
		COCA_news_uni_F_log = ktk.dict_builder(COCA_news_uni_list, 2,"y")

	#COCA Lemma Frequency and Range Lists
		
		dataQueue.put("Loading COCA Lemma Data... 1 of 4")
		root.update_idletasks()
		COCA_lemma_acad_uni_list = file(resource_path('COCA_acad_word_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_uni_R = ktk.dict_builder(COCA_lemma_acad_uni_list, 4)
		COCA_lemma_acad_uni_F = ktk.dict_builder(COCA_lemma_acad_uni_list, 2)
		COCA_lemma_acad_uni_R_log = ktk.dict_builder(COCA_lemma_acad_uni_list, 4,"y")
		COCA_lemma_acad_uni_F_log = ktk.dict_builder(COCA_lemma_acad_uni_list, 2,"y")

		COCA_lemma_acad_bi_list = file(resource_path('COCA_acad_bi_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_bi_R = ktk.dict_builder(COCA_lemma_acad_bi_list, 4)
		COCA_lemma_acad_bi_F = ktk.dict_builder(COCA_lemma_acad_bi_list, 2)

		COCA_lemma_acad_tri_list = file(resource_path('COCA_acad_tri_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_tri_R = ktk.dict_builder(COCA_lemma_acad_tri_list, 4)
		COCA_lemma_acad_tri_F = ktk.dict_builder(COCA_lemma_acad_tri_list, 2)

		COCA_lemma_acad_quad_list = file(resource_path('COCA_acad_quad_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_quad_R = ktk.dict_builder(COCA_lemma_acad_quad_list, 4)
		COCA_lemma_acad_quad_F = ktk.dict_builder(COCA_lemma_acad_quad_list, 2)

		#noun
		COCA_lemma_acad_n_bi_list = file(resource_path('COCA_acad_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_n_bi_R = ktk.dict_builder(COCA_lemma_acad_n_bi_list, 4)
		COCA_lemma_acad_n_bi_F = ktk.dict_builder(COCA_lemma_acad_n_bi_list, 2)

		COCA_lemma_acad_n_tri_list = file(resource_path('COCA_acad_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_n_tri_R = ktk.dict_builder(COCA_lemma_acad_n_tri_list, 4)
		COCA_lemma_acad_n_tri_F = ktk.dict_builder(COCA_lemma_acad_n_tri_list, 2)

		COCA_lemma_acad_n_quad_list = file(resource_path('COCA_acad_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_n_quad_R = ktk.dict_builder(COCA_lemma_acad_n_quad_list, 4)
		COCA_lemma_acad_n_quad_F = ktk.dict_builder(COCA_lemma_acad_n_quad_list, 2)

		#adjective
		COCA_lemma_acad_adj_bi_list = file(resource_path('COCA_acad_adj_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_adj_bi_R = ktk.dict_builder(COCA_lemma_acad_adj_bi_list, 4)
		COCA_lemma_acad_adj_bi_F = ktk.dict_builder(COCA_lemma_acad_adj_bi_list, 2)

		COCA_lemma_acad_adj_tri_list = file(resource_path('COCA_acad_adj_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_adj_tri_R = ktk.dict_builder(COCA_lemma_acad_adj_tri_list, 4)
		COCA_lemma_acad_adj_tri_F = ktk.dict_builder(COCA_lemma_acad_adj_tri_list, 2)

		COCA_lemma_acad_adj_quad_list = file(resource_path('COCA_acad_adj_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_adj_quad_R = ktk.dict_builder(COCA_lemma_acad_adj_quad_list, 4)
		COCA_lemma_acad_adj_quad_F = ktk.dict_builder(COCA_lemma_acad_adj_quad_list, 2)

		#verb
		COCA_lemma_acad_v_bi_list = file(resource_path('COCA_acad_v_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_v_bi_R = ktk.dict_builder(COCA_lemma_acad_v_bi_list, 4)
		COCA_lemma_acad_v_bi_F = ktk.dict_builder(COCA_lemma_acad_v_bi_list, 2)

		COCA_lemma_acad_v_tri_list = file(resource_path('COCA_acad_v_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_v_tri_R = ktk.dict_builder(COCA_lemma_acad_v_tri_list, 4)
		COCA_lemma_acad_v_tri_F = ktk.dict_builder(COCA_lemma_acad_v_tri_list, 2)

		COCA_lemma_acad_v_quad_list = file(resource_path('COCA_acad_v_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_v_quad_R = ktk.dict_builder(COCA_lemma_acad_v_quad_list, 4)
		COCA_lemma_acad_v_quad_F = ktk.dict_builder(COCA_lemma_acad_v_quad_list, 2)

		#verb_noun
		COCA_lemma_acad_v_n_bi_list = file(resource_path('COCA_acad_v_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_v_n_bi_R = ktk.dict_builder(COCA_lemma_acad_v_n_bi_list, 4)
		COCA_lemma_acad_v_n_bi_F = ktk.dict_builder(COCA_lemma_acad_v_n_bi_list, 2)

		COCA_lemma_acad_v_n_tri_list = file(resource_path('COCA_acad_v_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_v_n_tri_R = ktk.dict_builder(COCA_lemma_acad_v_n_tri_list, 4)
		COCA_lemma_acad_v_n_tri_F = ktk.dict_builder(COCA_lemma_acad_v_n_tri_list, 2)

		COCA_lemma_acad_v_n_quad_list = file(resource_path('COCA_acad_v_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_v_n_quad_R = ktk.dict_builder(COCA_lemma_acad_v_n_quad_list, 4)
		COCA_lemma_acad_v_n_quad_F = ktk.dict_builder(COCA_lemma_acad_v_n_quad_list, 2)

		#adjective_noun
		COCA_lemma_acad_a_n_bi_list = file(resource_path('COCA_acad_a_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_a_n_bi_R = ktk.dict_builder(COCA_lemma_acad_a_n_bi_list, 4)
		COCA_lemma_acad_a_n_bi_F = ktk.dict_builder(COCA_lemma_acad_a_n_bi_list, 2)

		COCA_lemma_acad_a_n_tri_list = file(resource_path('COCA_acad_a_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_a_n_tri_R = ktk.dict_builder(COCA_lemma_acad_a_n_tri_list, 4)
		COCA_lemma_acad_a_n_tri_F = ktk.dict_builder(COCA_lemma_acad_a_n_tri_list, 2)

		COCA_lemma_acad_a_n_quad_list = file(resource_path('COCA_acad_a_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_acad_a_n_quad_R = ktk.dict_builder(COCA_lemma_acad_a_n_quad_list, 4)
		COCA_lemma_acad_a_n_quad_F = ktk.dict_builder(COCA_lemma_acad_a_n_quad_list, 2)

	### Fiction
		dataQueue.put("Loading COCA Lemma Data... 2 of 4")
		root.update_idletasks()

		COCA_lemma_fic_uni_list = file(resource_path('COCA_fic_word_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_uni_R = ktk.dict_builder(COCA_lemma_fic_uni_list, 4)
		COCA_lemma_fic_uni_F = ktk.dict_builder(COCA_lemma_fic_uni_list, 2)
		COCA_lemma_fic_uni_R_log = ktk.dict_builder(COCA_lemma_fic_uni_list, 4,"y")
		COCA_lemma_fic_uni_F_log = ktk.dict_builder(COCA_lemma_fic_uni_list, 2,"y")

		COCA_lemma_fic_bi_list = file(resource_path('COCA_fic_bi_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_bi_R = ktk.dict_builder(COCA_lemma_fic_bi_list, 4)
		COCA_lemma_fic_bi_F = ktk.dict_builder(COCA_lemma_fic_bi_list, 2)

		COCA_lemma_fic_tri_list = file(resource_path('COCA_fic_tri_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_tri_R = ktk.dict_builder(COCA_lemma_fic_tri_list, 4)
		COCA_lemma_fic_tri_F = ktk.dict_builder(COCA_lemma_fic_tri_list, 2)

		COCA_lemma_fic_quad_list = file(resource_path('COCA_fic_quad_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_quad_R = ktk.dict_builder(COCA_lemma_fic_quad_list, 4)
		COCA_lemma_fic_quad_F = ktk.dict_builder(COCA_lemma_fic_quad_list, 2)

		#noun
		COCA_lemma_fic_n_bi_list = file(resource_path('COCA_fic_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_n_bi_R = ktk.dict_builder(COCA_lemma_fic_n_bi_list, 4)
		COCA_lemma_fic_n_bi_F = ktk.dict_builder(COCA_lemma_fic_n_bi_list, 2)

		COCA_lemma_fic_n_tri_list = file(resource_path('COCA_fic_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_n_tri_R = ktk.dict_builder(COCA_lemma_fic_n_tri_list, 4)
		COCA_lemma_fic_n_tri_F = ktk.dict_builder(COCA_lemma_fic_n_tri_list, 2)

		COCA_lemma_fic_n_quad_list = file(resource_path('COCA_fic_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_n_quad_R = ktk.dict_builder(COCA_lemma_fic_n_quad_list, 4)
		COCA_lemma_fic_n_quad_F = ktk.dict_builder(COCA_lemma_fic_n_quad_list, 2)

		#adjective
		COCA_lemma_fic_adj_bi_list = file(resource_path('COCA_fic_adj_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_adj_bi_R = ktk.dict_builder(COCA_lemma_fic_adj_bi_list, 4)
		COCA_lemma_fic_adj_bi_F = ktk.dict_builder(COCA_lemma_fic_adj_bi_list, 2)

		COCA_lemma_fic_adj_tri_list = file(resource_path('COCA_fic_adj_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_adj_tri_R = ktk.dict_builder(COCA_lemma_fic_adj_tri_list, 4)
		COCA_lemma_fic_adj_tri_F = ktk.dict_builder(COCA_lemma_fic_adj_tri_list, 2)

		COCA_lemma_fic_adj_quad_list = file(resource_path('COCA_fic_adj_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_adj_quad_R = ktk.dict_builder(COCA_lemma_fic_adj_quad_list, 4)
		COCA_lemma_fic_adj_quad_F = ktk.dict_builder(COCA_lemma_fic_adj_quad_list, 2)

		#verb
		COCA_lemma_fic_v_bi_list = file(resource_path('COCA_fic_v_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_v_bi_R = ktk.dict_builder(COCA_lemma_fic_v_bi_list, 4)
		COCA_lemma_fic_v_bi_F = ktk.dict_builder(COCA_lemma_fic_v_bi_list, 2)

		COCA_lemma_fic_v_tri_list = file(resource_path('COCA_fic_v_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_v_tri_R = ktk.dict_builder(COCA_lemma_fic_v_tri_list, 4)
		COCA_lemma_fic_v_tri_F = ktk.dict_builder(COCA_lemma_fic_v_tri_list, 2)

		COCA_lemma_fic_v_quad_list = file(resource_path('COCA_fic_v_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_v_quad_R = ktk.dict_builder(COCA_lemma_fic_v_quad_list, 4)
		COCA_lemma_fic_v_quad_F = ktk.dict_builder(COCA_lemma_fic_v_quad_list, 2)

		#verb_noun
		COCA_lemma_fic_v_n_bi_list = file(resource_path('COCA_fic_v_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_v_n_bi_R = ktk.dict_builder(COCA_lemma_fic_v_n_bi_list, 4)
		COCA_lemma_fic_v_n_bi_F = ktk.dict_builder(COCA_lemma_fic_v_n_bi_list, 2)

		COCA_lemma_fic_v_n_tri_list = file(resource_path('COCA_fic_v_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_v_n_tri_R = ktk.dict_builder(COCA_lemma_fic_v_n_tri_list, 4)
		COCA_lemma_fic_v_n_tri_F = ktk.dict_builder(COCA_lemma_fic_v_n_tri_list, 2)

		COCA_lemma_fic_v_n_quad_list = file(resource_path('COCA_fic_v_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_v_n_quad_R = ktk.dict_builder(COCA_lemma_fic_v_n_quad_list, 4)
		COCA_lemma_fic_v_n_quad_F = ktk.dict_builder(COCA_lemma_fic_v_n_quad_list, 2)

		#adjective_noun
		COCA_lemma_fic_a_n_bi_list = file(resource_path('COCA_fic_a_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_a_n_bi_R = ktk.dict_builder(COCA_lemma_fic_a_n_bi_list, 4)
		COCA_lemma_fic_a_n_bi_F = ktk.dict_builder(COCA_lemma_fic_a_n_bi_list, 2)

		COCA_lemma_fic_a_n_tri_list = file(resource_path('COCA_fic_a_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_a_n_tri_R = ktk.dict_builder(COCA_lemma_fic_a_n_tri_list, 4)
		COCA_lemma_fic_a_n_tri_F = ktk.dict_builder(COCA_lemma_fic_a_n_tri_list, 2)

		COCA_lemma_fic_a_n_quad_list = file(resource_path('COCA_fic_a_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_fic_a_n_quad_R = ktk.dict_builder(COCA_lemma_fic_a_n_quad_list, 4)
		COCA_lemma_fic_a_n_quad_F = ktk.dict_builder(COCA_lemma_fic_a_n_quad_list, 2)
	
	### Magazine
		dataQueue.put("Loading COCA Lemma Data... 3 of 4")
		root.update_idletasks()

		COCA_lemma_mag_uni_list = file(resource_path('COCA_mag_word_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_uni_R = ktk.dict_builder(COCA_lemma_mag_uni_list, 4)
		COCA_lemma_mag_uni_F = ktk.dict_builder(COCA_lemma_mag_uni_list, 2)
		COCA_lemma_mag_uni_R_log = ktk.dict_builder(COCA_lemma_mag_uni_list, 4,"y")
		COCA_lemma_mag_uni_F_log = ktk.dict_builder(COCA_lemma_mag_uni_list, 2,"y")

		COCA_lemma_mag_bi_list = file(resource_path('COCA_mag_bi_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_bi_R = ktk.dict_builder(COCA_lemma_mag_bi_list, 4)
		COCA_lemma_mag_bi_F = ktk.dict_builder(COCA_lemma_mag_bi_list, 2)

		COCA_lemma_mag_tri_list = file(resource_path('COCA_mag_tri_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_tri_R = ktk.dict_builder(COCA_lemma_mag_tri_list, 4)
		COCA_lemma_mag_tri_F = ktk.dict_builder(COCA_lemma_mag_tri_list, 2)

		COCA_lemma_mag_quad_list = file(resource_path('COCA_mag_quad_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_quad_R = ktk.dict_builder(COCA_lemma_mag_quad_list, 4)
		COCA_lemma_mag_quad_F = ktk.dict_builder(COCA_lemma_mag_quad_list, 2)

		#noun
		COCA_lemma_mag_n_bi_list = file(resource_path('COCA_mag_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_n_bi_R = ktk.dict_builder(COCA_lemma_mag_n_bi_list, 4)
		COCA_lemma_mag_n_bi_F = ktk.dict_builder(COCA_lemma_mag_n_bi_list, 2)

		COCA_lemma_mag_n_tri_list = file(resource_path('COCA_mag_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_n_tri_R = ktk.dict_builder(COCA_lemma_mag_n_tri_list, 4)
		COCA_lemma_mag_n_tri_F = ktk.dict_builder(COCA_lemma_mag_n_tri_list, 2)

		COCA_lemma_mag_n_quad_list = file(resource_path('COCA_mag_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_n_quad_R = ktk.dict_builder(COCA_lemma_mag_n_quad_list, 4)
		COCA_lemma_mag_n_quad_F = ktk.dict_builder(COCA_lemma_mag_n_quad_list, 2)

		#adjective
		COCA_lemma_mag_adj_bi_list = file(resource_path('COCA_mag_adj_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_adj_bi_R = ktk.dict_builder(COCA_lemma_mag_adj_bi_list, 4)
		COCA_lemma_mag_adj_bi_F = ktk.dict_builder(COCA_lemma_mag_adj_bi_list, 2)

		COCA_lemma_mag_adj_tri_list = file(resource_path('COCA_mag_adj_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_adj_tri_R = ktk.dict_builder(COCA_lemma_mag_adj_tri_list, 4)
		COCA_lemma_mag_adj_tri_F = ktk.dict_builder(COCA_lemma_mag_adj_tri_list, 2)

		COCA_lemma_mag_adj_quad_list = file(resource_path('COCA_mag_adj_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_adj_quad_R = ktk.dict_builder(COCA_lemma_mag_adj_quad_list, 4)
		COCA_lemma_mag_adj_quad_F = ktk.dict_builder(COCA_lemma_mag_adj_quad_list, 2)

		#verb
		COCA_lemma_mag_v_bi_list = file(resource_path('COCA_mag_v_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_v_bi_R = ktk.dict_builder(COCA_lemma_mag_v_bi_list, 4)
		COCA_lemma_mag_v_bi_F = ktk.dict_builder(COCA_lemma_mag_v_bi_list, 2)

		COCA_lemma_mag_v_tri_list = file(resource_path('COCA_mag_v_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_v_tri_R = ktk.dict_builder(COCA_lemma_mag_v_tri_list, 4)
		COCA_lemma_mag_v_tri_F = ktk.dict_builder(COCA_lemma_mag_v_tri_list, 2)

		COCA_lemma_mag_v_quad_list = file(resource_path('COCA_mag_v_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_v_quad_R = ktk.dict_builder(COCA_lemma_mag_v_quad_list, 4)
		COCA_lemma_mag_v_quad_F = ktk.dict_builder(COCA_lemma_mag_v_quad_list, 2)

		#verb_noun
		COCA_lemma_mag_v_n_bi_list = file(resource_path('COCA_mag_v_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_v_n_bi_R = ktk.dict_builder(COCA_lemma_mag_v_n_bi_list, 4)
		COCA_lemma_mag_v_n_bi_F = ktk.dict_builder(COCA_lemma_mag_v_n_bi_list, 2)

		COCA_lemma_mag_v_n_tri_list = file(resource_path('COCA_mag_v_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_v_n_tri_R = ktk.dict_builder(COCA_lemma_mag_v_n_tri_list, 4)
		COCA_lemma_mag_v_n_tri_F = ktk.dict_builder(COCA_lemma_mag_v_n_tri_list, 2)

		COCA_lemma_mag_v_n_quad_list = file(resource_path('COCA_mag_v_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_v_n_quad_R = ktk.dict_builder(COCA_lemma_mag_v_n_quad_list, 4)
		COCA_lemma_mag_v_n_quad_F = ktk.dict_builder(COCA_lemma_mag_v_n_quad_list, 2)

		#adjective_noun
		COCA_lemma_mag_a_n_bi_list = file(resource_path('COCA_mag_a_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_a_n_bi_R = ktk.dict_builder(COCA_lemma_mag_a_n_bi_list, 4)
		COCA_lemma_mag_a_n_bi_F = ktk.dict_builder(COCA_lemma_mag_a_n_bi_list, 2)

		COCA_lemma_mag_a_n_tri_list = file(resource_path('COCA_mag_a_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_a_n_tri_R = ktk.dict_builder(COCA_lemma_mag_a_n_tri_list, 4)
		COCA_lemma_mag_a_n_tri_F = ktk.dict_builder(COCA_lemma_mag_a_n_tri_list, 2)

		COCA_lemma_mag_a_n_quad_list = file(resource_path('COCA_mag_a_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_mag_a_n_quad_R = ktk.dict_builder(COCA_lemma_mag_a_n_quad_list, 4)
		COCA_lemma_mag_a_n_quad_F = ktk.dict_builder(COCA_lemma_mag_a_n_quad_list, 2)
	
	### News
		dataQueue.put("Loading COCA Lemma Data... 4 of 4")
		root.update_idletasks()

		COCA_lemma_news_uni_list = file(resource_path('COCA_news_word_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_uni_R = ktk.dict_builder(COCA_lemma_news_uni_list, 4)
		COCA_lemma_news_uni_F = ktk.dict_builder(COCA_lemma_news_uni_list, 2)
		COCA_lemma_news_uni_R_log = ktk.dict_builder(COCA_lemma_news_uni_list, 4,"y")
		COCA_lemma_news_uni_F_log = ktk.dict_builder(COCA_lemma_news_uni_list, 2,"y")

		COCA_lemma_news_bi_list = file(resource_path('COCA_news_bi_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_bi_R = ktk.dict_builder(COCA_lemma_news_bi_list, 4)
		COCA_lemma_news_bi_F = ktk.dict_builder(COCA_lemma_news_bi_list, 2)

		COCA_lemma_news_tri_list = file(resource_path('COCA_news_tri_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_tri_R = ktk.dict_builder(COCA_lemma_news_tri_list, 4)
		COCA_lemma_news_tri_F = ktk.dict_builder(COCA_lemma_news_tri_list, 2)

		COCA_lemma_news_quad_list = file(resource_path('COCA_news_quad_list_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_quad_R = ktk.dict_builder(COCA_lemma_news_quad_list, 4)
		COCA_lemma_news_quad_F = ktk.dict_builder(COCA_lemma_news_quad_list, 2)

		#noun
		COCA_lemma_news_n_bi_list = file(resource_path('COCA_news_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_n_bi_R = ktk.dict_builder(COCA_lemma_news_n_bi_list, 4)
		COCA_lemma_news_n_bi_F = ktk.dict_builder(COCA_lemma_news_n_bi_list, 2)

		COCA_lemma_news_n_tri_list = file(resource_path('COCA_news_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_n_tri_R = ktk.dict_builder(COCA_lemma_news_n_tri_list, 4)
		COCA_lemma_news_n_tri_F = ktk.dict_builder(COCA_lemma_news_n_tri_list, 2)

		COCA_lemma_news_n_quad_list = file(resource_path('COCA_news_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_n_quad_R = ktk.dict_builder(COCA_lemma_news_n_quad_list, 4)
		COCA_lemma_news_n_quad_F = ktk.dict_builder(COCA_lemma_news_n_quad_list, 2)

		#adjective
		COCA_lemma_news_adj_bi_list = file(resource_path('COCA_news_adj_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_adj_bi_R = ktk.dict_builder(COCA_lemma_news_adj_bi_list, 4)
		COCA_lemma_news_adj_bi_F = ktk.dict_builder(COCA_lemma_news_adj_bi_list, 2)

		COCA_lemma_news_adj_tri_list = file(resource_path('COCA_news_adj_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_adj_tri_R = ktk.dict_builder(COCA_lemma_news_adj_tri_list, 4)
		COCA_lemma_news_adj_tri_F = ktk.dict_builder(COCA_lemma_news_adj_tri_list, 2)

		COCA_lemma_news_adj_quad_list = file(resource_path('COCA_news_adj_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_adj_quad_R = ktk.dict_builder(COCA_lemma_news_adj_quad_list, 4)
		COCA_lemma_news_adj_quad_F = ktk.dict_builder(COCA_lemma_news_adj_quad_list, 2)

		#verb
		COCA_lemma_news_v_bi_list = file(resource_path('COCA_news_v_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_v_bi_R = ktk.dict_builder(COCA_lemma_news_v_bi_list, 4)
		COCA_lemma_news_v_bi_F = ktk.dict_builder(COCA_lemma_news_v_bi_list, 2)

		COCA_lemma_news_v_tri_list = file(resource_path('COCA_news_v_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_v_tri_R = ktk.dict_builder(COCA_lemma_news_v_tri_list, 4)
		COCA_lemma_news_v_tri_F = ktk.dict_builder(COCA_lemma_news_v_tri_list, 2)

		COCA_lemma_news_v_quad_list = file(resource_path('COCA_news_v_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_v_quad_R = ktk.dict_builder(COCA_lemma_news_v_quad_list, 4)
		COCA_lemma_news_v_quad_F = ktk.dict_builder(COCA_lemma_news_v_quad_list, 2)

		#verb_noun
		COCA_lemma_news_v_n_bi_list = file(resource_path('COCA_news_v_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_v_n_bi_R = ktk.dict_builder(COCA_lemma_news_v_n_bi_list, 4)
		COCA_lemma_news_v_n_bi_F = ktk.dict_builder(COCA_lemma_news_v_n_bi_list, 2)

		COCA_lemma_news_v_n_tri_list = file(resource_path('COCA_news_v_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_v_n_tri_R = ktk.dict_builder(COCA_lemma_news_v_n_tri_list, 4)
		COCA_lemma_news_v_n_tri_F = ktk.dict_builder(COCA_lemma_news_v_n_tri_list, 2)

		COCA_lemma_news_v_n_quad_list = file(resource_path('COCA_news_v_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_v_n_quad_R = ktk.dict_builder(COCA_lemma_news_v_n_quad_list, 4)
		COCA_lemma_news_v_n_quad_F = ktk.dict_builder(COCA_lemma_news_v_n_quad_list, 2)

		#adjective_noun
		COCA_lemma_news_a_n_bi_list = file(resource_path('COCA_news_a_n_list_bi_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_a_n_bi_R = ktk.dict_builder(COCA_lemma_news_a_n_bi_list, 4)
		COCA_lemma_news_a_n_bi_F = ktk.dict_builder(COCA_lemma_news_a_n_bi_list, 2)

		COCA_lemma_news_a_n_tri_list = file(resource_path('COCA_news_a_n_list_tri_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_a_n_tri_R = ktk.dict_builder(COCA_lemma_news_a_n_tri_list, 4)
		COCA_lemma_news_a_n_tri_F = ktk.dict_builder(COCA_lemma_news_a_n_tri_list, 2)

		COCA_lemma_news_a_n_quad_list = file(resource_path('COCA_news_a_n_list_quad_lemma_freq.csv'), 'rU').read()
		COCA_lemma_news_a_n_quad_R = ktk.dict_builder(COCA_lemma_news_a_n_quad_list, 4)
		COCA_lemma_news_a_n_quad_F = ktk.dict_builder(COCA_lemma_news_a_n_quad_list, 2)

	#BNC Program Files
		dataQueue.put("Loading BNC Word Data...")
		freq_list_written = file(resource_path('bnc_written.txt'), 'rU').read()
		bnc_freq_written = ktk.dict_builder(freq_list_written, 2)
		bnc_freq_written_log = ktk.dict_builder(freq_list_written, 2,"y")
		bnc_range_written = ktk.dict_builder(freq_list_written, 4)
					
		dataQueue.put("Loading LSA Matrix...")
		root.update_idletasks()
		lsa_matrix_list = file(resource_path('tasa_lsa_matrix.txt'), 'rU').read()
		
		dataQueue.put("Loading LSA Weights...")		
		root.update_idletasks()
		lsa_weights_list = file(resource_path('lsa_weights.txt'), 'rU').read()

		dataQueue.put("Loading LSA Matrix Dict (be patient) ...")
		root.update_idletasks()
		lsa_matrix = ktk.list_dict_builder(lsa_matrix_list,numbers="yes")

		dataQueue.put("Loading LSA Weight Dict...")
		root.update_idletasks()
		lsa_weights = ktk.dict_builder(lsa_weights_list,2)
		
	#Wordnet Databases
		dataQueue.put("Loading Wordnet Databases...")
		root.update_idletasks()
		wn_verb_list = file(resource_path('wn_verb.txt'), 'rU').read()
		wn_verb_dict = ktk.list_dict_builder(wn_verb_list)
		wn_noun_list = file(resource_path('wn_noun.txt'), 'rU').read()
		wn_noun_dict = ktk.list_dict_builder(wn_noun_list)
		
	#### Source Summary Analysis ####
		dataQueue.put("Processing Summary Text..")
		root.update_idletasks()

		source = file(summ_text,"rU").read().lower()
		
		npar_source = ktk.n_paragraphs(source)
		
		source_clean = ktk.text_cleaner(source) #text now string of words
		
		#print "Source clean: ", source_clean
		nwords_source = len(source_clean)
		
		tagged_source_name = "parsed_files/" + summ_text.split("/")[-1] + ".xml"
		
		dataQueue.put("Loading Summary Text POS Lists")

		source_pos_dict = ktk.content_pos_dict(tagged_source_name) #dict of pos lists
		source_pos_lem_dict = ktk.content_pos_dict(tagged_source_name,lemma = "yes") #dict of pos lists

		dataQueue.put("Loading Summary Text Ngram Lists")

		source_ngram_pos_dict = ktk.ngram_pos_dict(tagged_source_name)
		source_ngram_pos_lem_dict = ktk.ngram_pos_dict(tagged_source_name, lemma = "yes")

	## More Source analysis ##
		#model for keyword lists:
		simple_source_keywords = ktk.keyness(source_clean, COCA_fiction_uni_F, top_perc = .1)
		
		acad_keywords = ktk.keyness(source_pos_lem_dict["all"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_keywords.txt")
		acad_n_keywords = ktk.keyness(source_pos_lem_dict["noun"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_n_keywords.txt")
		acad_no_pn_nkeywords = ktk.keyness(source_pos_lem_dict["no_proper"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_no_pn_nkeywords.txt")
		acad_pn_keywords = ktk.keyness(source_pos_lem_dict["proper_n"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_pn_keywords.txt")
		acad_v_keywords = ktk.keyness(source_pos_lem_dict["verb"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_keywords.txt")
		acad_v_n_keywords = ktk.keyness(source_pos_lem_dict["verb_noun"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_n_keywords.txt")
		acad_adj_keywords = ktk.keyness(source_pos_lem_dict["adj"], COCA_lemma_acad_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_adj_keywords.txt")
		
		fic_keywords = ktk.keyness(source_pos_lem_dict["all"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_keywords.txt")
		fic_n_keywords = ktk.keyness(source_pos_lem_dict["noun"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_n_keywords.txt")
		fic_no_pn_nkeywords = ktk.keyness(source_pos_lem_dict["no_proper"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_no_pn_nkeywords.txt")
		fic_pn_keywords = ktk.keyness(source_pos_lem_dict["proper_n"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_pn_keywords.txt")
		fic_v_keywords = ktk.keyness(source_pos_lem_dict["verb"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_keywords.txt")
		fic_v_n_keywords = ktk.keyness(source_pos_lem_dict["verb_noun"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_n_keywords.txt")
		fic_adj_keywords = ktk.keyness(source_pos_lem_dict["adj"], COCA_lemma_fic_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_adj_keywords.txt")
		
		mag_keywords = ktk.keyness(source_pos_lem_dict["all"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_keywords.txt")
		mag_n_keywords = ktk.keyness(source_pos_lem_dict["noun"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_n_keywords.txt")
		mag_no_pn_nkeywords = ktk.keyness(source_pos_lem_dict["no_proper"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_no_pn_nkeywords.txt")
		mag_pn_keywords = ktk.keyness(source_pos_lem_dict["proper_n"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_pn_keywords.txt")
		mag_v_keywords = ktk.keyness(source_pos_lem_dict["verb"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_keywords.txt")
		mag_v_n_keywords = ktk.keyness(source_pos_lem_dict["verb_noun"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_n_keywords.txt")
		mag_adj_keywords = ktk.keyness(source_pos_lem_dict["adj"], COCA_lemma_mag_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_adj_keywords.txt")
		
		news_keywords = ktk.keyness(source_pos_lem_dict["all"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_keywords.txt")
		news_n_keywords = ktk.keyness(source_pos_lem_dict["noun"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_n_keywords.txt")
		news_no_pn_nkeywords = ktk.keyness(source_pos_lem_dict["no_proper"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_no_pn_nkeywords.txt")
		news_pn_keywords = ktk.keyness(source_pos_lem_dict["proper_n"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_pn_keywords.txt")
		news_v_keywords = ktk.keyness(source_pos_lem_dict["verb"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_keywords.txt")
		news_v_n_keywords = ktk.keyness(source_pos_lem_dict["verb_noun"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_n_keywords.txt")
		news_adj_keywords = ktk.keyness(source_pos_lem_dict["adj"], COCA_lemma_news_uni_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_adj_keywords.txt")
		
		acad_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["bi_list"], COCA_lemma_acad_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_bi_keywords.txt")
		acad_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["tri_list"], COCA_lemma_acad_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_tri_keywords.txt")
		acad_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["quad_list"], COCA_lemma_acad_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_quad_keywords.txt")
		
		acad_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_bi"], COCA_lemma_acad_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_n_bi_keywords.txt")
		acad_adj_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_bi"], COCA_lemma_acad_adj_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_adj_bi_keywords.txt")
		acad_v_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_bi"], COCA_lemma_acad_v_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_bi_keywords.txt")
		acad_v_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_bi"], COCA_lemma_acad_v_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_n_bi_keywords.txt")
		acad_a_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_bi"], COCA_lemma_acad_a_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_a_n_bi_keywords.txt")
		acad_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_tri"], COCA_lemma_acad_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_n_tri_keywords.txt")
		acad_adj_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_tri"], COCA_lemma_acad_adj_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_adj_tri_keywords.txt")
		acad_v_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_tri"], COCA_lemma_acad_v_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_tri_keywords.txt")
		acad_v_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_tri"], COCA_lemma_acad_v_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_n_tri_keywords.txt")
		acad_a_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_tri"], COCA_lemma_acad_a_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_a_n_tri_keywords.txt")
		acad_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_quad"], COCA_lemma_acad_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_n_quad_keywords.txt")
		acad_adj_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_quad"], COCA_lemma_acad_adj_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_adj_quad_keywords.txt")
		acad_v_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_quad"], COCA_lemma_acad_v_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_quad_keywords.txt")
		acad_v_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_quad"], COCA_lemma_acad_v_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_v_n_quad_keywords.txt")
		acad_a_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_quad"], COCA_lemma_acad_a_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "acad_a_n_quad_keywords.txt")
		
		fic_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["bi_list"], COCA_lemma_fic_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_bi_keywords.txt")
		fic_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["tri_list"], COCA_lemma_fic_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_tri_keywords.txt")
		fic_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["quad_list"], COCA_lemma_fic_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_quad_keywords.txt")
		fic_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_bi"], COCA_lemma_fic_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_n_bi_keywords.txt")
		fic_adj_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_bi"], COCA_lemma_fic_adj_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_adj_bi_keywords.txt")
		fic_v_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_bi"], COCA_lemma_fic_v_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_bi_keywords.txt")
		fic_v_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_bi"], COCA_lemma_fic_v_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_n_bi_keywords.txt")
		fic_a_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_bi"], COCA_lemma_fic_a_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_a_n_bi_keywords.txt")
		fic_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_tri"], COCA_lemma_fic_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_n_tri_keywords.txt")
		fic_adj_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_tri"], COCA_lemma_fic_adj_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_adj_tri_keywords.txt")
		fic_v_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_tri"], COCA_lemma_fic_v_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_tri_keywords.txt")
		fic_v_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_tri"], COCA_lemma_fic_v_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_n_tri_keywords.txt")
		fic_a_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_tri"], COCA_lemma_fic_a_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_a_n_tri_keywords.txt")
		fic_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_quad"], COCA_lemma_fic_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_n_quad_keywords.txt")
		fic_adj_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_quad"], COCA_lemma_fic_adj_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_adj_quad_keywords.txt")
		fic_v_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_quad"], COCA_lemma_fic_v_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_quad_keywords.txt")
		fic_v_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_quad"], COCA_lemma_fic_v_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_v_n_quad_keywords.txt")
		fic_a_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_quad"], COCA_lemma_fic_a_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "fic_a_n_quad_keywords.txt")
		
		mag_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["bi_list"], COCA_lemma_mag_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_bi_keywords.txt")
		mag_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["tri_list"], COCA_lemma_mag_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_tri_keywords.txt")
		mag_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["quad_list"], COCA_lemma_mag_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_quad_keywords.txt")
		mag_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_bi"], COCA_lemma_mag_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_n_bi_keywords.txt")
		mag_adj_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_bi"], COCA_lemma_mag_adj_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_adj_bi_keywords.txt")
		mag_v_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_bi"], COCA_lemma_mag_v_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_bi_keywords.txt")
		mag_v_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_bi"], COCA_lemma_mag_v_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_n_bi_keywords.txt")
		mag_a_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_bi"], COCA_lemma_mag_a_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_a_n_bi_keywords.txt")
		mag_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_tri"], COCA_lemma_mag_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_n_tri_keywords.txt")
		mag_adj_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_tri"], COCA_lemma_mag_adj_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_adj_tri_keywords.txt")
		mag_v_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_tri"], COCA_lemma_mag_v_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_tri_keywords.txt")
		mag_v_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_tri"], COCA_lemma_mag_v_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_n_tri_keywords.txt")
		mag_a_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_tri"], COCA_lemma_mag_a_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_a_n_tri_keywords.txt")
		mag_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_quad"], COCA_lemma_mag_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_n_quad_keywords.txt")
		mag_adj_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_quad"], COCA_lemma_mag_adj_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_adj_quad_keywords.txt")
		mag_v_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_quad"], COCA_lemma_mag_v_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_quad_keywords.txt")
		mag_v_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_quad"], COCA_lemma_mag_v_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_v_n_quad_keywords.txt")
		mag_a_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_quad"], COCA_lemma_mag_a_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "mag_a_n_quad_keywords.txt")
		
		news_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["bi_list"], COCA_lemma_news_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_bi_keywords.txt")
		news_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["tri_list"], COCA_lemma_news_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_tri_keywords.txt")
		news_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["quad_list"], COCA_lemma_news_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_quad_keywords.txt")
		news_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_bi"], COCA_lemma_news_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_n_bi_keywords.txt")
		news_adj_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_bi"], COCA_lemma_news_adj_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_adj_bi_keywords.txt")
		news_v_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_bi"], COCA_lemma_news_v_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_bi_keywords.txt")
		news_v_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_bi"], COCA_lemma_news_v_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_n_bi_keywords.txt")
		news_a_n_bi_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_bi"], COCA_lemma_news_a_n_bi_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_a_n_bi_keywords.txt")
		news_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_tri"], COCA_lemma_news_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_n_tri_keywords.txt")
		news_adj_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_tri"], COCA_lemma_news_adj_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_adj_tri_keywords.txt")
		news_v_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_tri"], COCA_lemma_news_v_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_tri_keywords.txt")
		news_v_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_tri"], COCA_lemma_news_v_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_n_tri_keywords.txt")
		news_a_n_tri_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_tri"], COCA_lemma_news_a_n_tri_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_a_n_tri_keywords.txt")
		news_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["n_list_quad"], COCA_lemma_news_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_n_quad_keywords.txt")
		news_adj_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["adj_list_quad"], COCA_lemma_news_adj_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_adj_quad_keywords.txt")
		news_v_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_list_quad"], COCA_lemma_news_v_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_quad_keywords.txt")
		news_v_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["v_n_list_quad"], COCA_lemma_news_v_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_v_n_quad_keywords.txt")
		news_a_n_quad_keywords = ktk.keyness(source_ngram_pos_lem_dict["a_n_list_quad"], COCA_lemma_news_a_n_quad_F, top_perc = .1,out_dir = key_out_dir,keyname = "news_a_n_quad_keywords.txt")
	### End Source Summary Analysis
		
		nfiles = len(filenames)
		
	
	### Begin Iteration Through Summary Responses ###
		for filename in filenames:
			header_list = ["Filename"]
			index_list = []
			
		#updates Program Status
			filename1 = ("Processing: " + str(file_counter) + " of " + str(nfiles) + " files")
			dataQueue.put(filename1)
			root.update_idletasks()
			file_counter+=1
			
			text= file(filename, 'rU').read().lower()
		
			clean_text = ktk.text_cleaner(text)
			coca_text = ktk.coca_texter(clean_text)
			
			nwords_text = len(clean_text)
			npar_text = ktk.n_paragraphs(text)			
			ktk.indexer(nwords_text, "nwords", index_list, header_list)
			ktk.indexer(npar_text, "nparagraphs", index_list, header_list)
			tagged_text_name = "parsed_files/" + filename.split("/")[-1] + ".xml"
	
			pos_dict = ktk.content_pos_dict(tagged_text_name)
			pos_lem_dict = ktk.content_pos_dict(tagged_text_name,lemma = "yes")
			ngram_pos_dict = ktk.ngram_pos_dict(tagged_text_name)
			ngram_pos_lem_dict = ktk.ngram_pos_dict(tagged_text_name, lemma = "yes")

			parsed_nwords = len(pos_dict["all"])
			ktk.indexer(parsed_nwords, "parsed_nwords",index_list,header_list)
		
			ktk.indexer(len(pos_dict["s_all"]),"nsentences",index_list,header_list)
			
			ktk.indexer(ktk.safe_divide(len(pos_dict["content"]),parsed_nwords),"content_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["function"]),parsed_nwords),"function_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["noun"]),parsed_nwords),"all_noun_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["proper_n"]),parsed_nwords),"proper_n_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["no_proper"]),parsed_nwords),"no_proper_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["pronoun"]),parsed_nwords),"pronoun_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["verb"]),parsed_nwords),"verb_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["adj"]),parsed_nwords),"adj_perc",index_list,header_list)
			ktk.indexer(ktk.safe_divide(len(pos_dict["adv"]),parsed_nwords),"adv_perc",index_list,header_list)
			#ktk.indexer(ktk.safe_divide(len(pos_dict["s_all"]),parsed_nwords),"nsentences_perc",index_list,header_list)
			
			
		#Psycholinguistic_Norms
			ktk.DataDict_counter(clean_text,fam,"all","0",index_list,"MRC_Familiarity_AW",header_list)
			ktk.DataDict_counter(clean_text,concreteness,"all","0",index_list,"MRC_Concreteness_AW",header_list)
			ktk.DataDict_counter(clean_text,imageability,"all","0",index_list,"MRC_Imageability_AW",header_list)
			ktk.DataDict_counter(clean_text,meaningfulness_colorado,"all","0",index_list,"MRC_Meaningfulness_AW",header_list)
			ktk.DataDict_counter(clean_text,fam,"cw","0",index_list,"MRC_Familiarity_CW",header_list)
			ktk.DataDict_counter(clean_text,concreteness,"cw","0",index_list,"MRC_Concreteness_CW",header_list)
			ktk.DataDict_counter(clean_text,imageability,"cw","0",index_list,"MRC_Imageability_CW",header_list)
			ktk.DataDict_counter(clean_text,fam,"fw","0",index_list,"MRC_Familiarity_FW",header_list)
			ktk.DataDict_counter(clean_text,concreteness,"fw","0",index_list,"MRC_Concreteness_FW",header_list)
			ktk.DataDict_counter(clean_text,imageability,"fw","0",index_list,"MRC_Imageability_FW",header_list)
			ktk.DataDict_counter(clean_text,meaningfulness_colorado,"fw","0",index_list,"MRC_Meaningfulness_FW",header_list)
			
		#_Kuperman_et_al._AoA_Mean_Scores:
			ktk.DataDict_counter(clean_text,B_AoA,"all","0",index_list,"Kuperman_AoA_AW",header_list)
			ktk.DataDict_counter(clean_text,B_AoA,"cw","0",index_list,"Kuperman_AoA_CW",header_list)
			ktk.DataDict_counter(clean_text,B_AoA,"fw","0",index_list,"Kuperman_AoA_FW",header_list)
			
		#Brysbaert_et_al_Unigram_Concreteness:
			ktk.Mixed_DataDict_counter(clean_text,B_Conc,"aw",index_list,"Brysbaert_Concreteness_Combined_AW",header_list)
			ktk.Mixed_DataDict_counter(clean_text,B_Conc,"cw",index_list,"Brysbaert_Concreteness_Combined_CW",header_list)
			ktk.Mixed_DataDict_counter(clean_text,B_Conc,"fw",index_list,"Brysbaert_Concreteness_Combined_FW",header_list)
			
		#Subtlexus
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.DataDict_counter(clean_text,subtlex_freq,"all","0",index_list,"SUBTLEXus_Freq_AW",header_list)
			ktk.DataDict_counter(clean_text,subtlex_cd,"all","0",index_list,"SUBTLEXus_Range_AW",header_list)
			ktk.DataDict_counter(clean_text,subtlex_freq_log,"all","0",index_list,"SUBTLEXus_Freq_AW_Log",header_list)
			ktk.DataDict_counter(clean_text,subtlex_cd_log,"all","0",index_list,"SUBTLEXus_Range_AW_Log",header_list)

			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):

			ktk.DataDict_counter(clean_text,subtlex_freq,"cw","0",index_list,"SUBTLEXus_Freq_CW",header_list)
			ktk.DataDict_counter(clean_text,subtlex_cd,"cw","0",index_list,"SUBTLEXus_Range_CW",header_list)
			ktk.DataDict_counter(clean_text,subtlex_freq_log,"cw","0",index_list,"SUBTLEXus_Freq_CW_Log",header_list)
			ktk.DataDict_counter(clean_text,subtlex_cd_log,"cw","0",index_list,"SUBTLEXus_Range_CW_Log",header_list)
		
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
		
			ktk.DataDict_counter(clean_text,subtlex_freq,"fw","0",index_list,"SUBTLEXus_Freq_FW",header_list)
			ktk.DataDict_counter(clean_text,subtlex_cd,"fw","0",index_list,"SUBTLEXus_Range_FW",header_list)
			ktk.DataDict_counter(clean_text,subtlex_freq_log,"fw","0",index_list,"SUBTLEXus_Freq_FW_Log",header_list)
			ktk.DataDict_counter(clean_text,subtlex_cd_log,"fw","0",index_list,"SUBTLEXus_Range_FW_Log",header_list)
		
		#BNC_Word
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.DataDict_counter(clean_text,bnc_freq_written,"all","0",index_list,"BNC_Written_Freq_AW",header_list)
			ktk.DataDict_counter(clean_text,bnc_freq_written_log,"all","0",index_list,"BNC_Written_Freq_AW_Log",header_list)
			ktk.DataDict_counter(clean_text,bnc_range_written,"all","0",index_list,"BNC_Written_Range_AW",header_list)
			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):
			ktk.DataDict_counter(clean_text,bnc_freq_written,"cw","0",index_list,"BNC_Written_Freq_CW",header_list)
			ktk.DataDict_counter(clean_text,bnc_freq_written_log,"cw","0",index_list,"BNC_Written_Freq_CW_Log",header_list)
			ktk.DataDict_counter(clean_text,bnc_range_written,"cw","0",index_list,"BNC_Written_Range_CW",header_list)
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
			ktk.DataDict_counter(clean_text,bnc_freq_written,"fw","0",index_list,"BNC_Written_Freq_FW",header_list)
			ktk.DataDict_counter(clean_text,bnc_freq_written_log,"fw","0",index_list,"BNC_Written_Freq_FW_Log",header_list)
			ktk.DataDict_counter(clean_text,bnc_range_written,"fw","0",index_list,"BNC_Written_Range_FW",header_list)

		#Psycholinguistic_Norms
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),fam,"all","0",index_list,"MRC_Familiarity_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),concreteness,"all","0",index_list,"MRC_Concreteness_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),imageability,"all","0",index_list,"MRC_Imageability_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),meaningfulness_colorado,"all","0",index_list,"MRC_Meaningfulness_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),fam,"cw","0",index_list,"MRC_Familiarity_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),concreteness,"cw","0",index_list,"MRC_Concreteness_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),imageability,"cw","0",index_list,"MRC_Imageability_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),fam,"fw","0",index_list,"MRC_Familiarity_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),concreteness,"fw","0",index_list,"MRC_Concreteness_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),imageability,"fw","0",index_list,"MRC_Imageability_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),meaningfulness_colorado,"fw","0",index_list,"MRC_Meaningfulness_FW_no_kw",header_list)
			
		#_Kuperman_et_al._AoA_Mean_Scores:
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),B_AoA,"all","0",index_list,"Kuperman_AoA_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),B_AoA,"cw","0",index_list,"Kuperman_AoA_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),B_AoA,"fw","0",index_list,"Kuperman_AoA_FW_no_kw",header_list)
			
		#Brysbaert_et_al_Unigram_Concreteness:
			ktk.Mixed_DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),B_Conc,"aw",index_list,"Brysbaert_Concreteness_Combined_AW_no_kw",header_list)
			ktk.Mixed_DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),B_Conc,"cw",index_list,"Brysbaert_Concreteness_Combined_CW_no_kw",header_list)
			ktk.Mixed_DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),B_Conc,"fw",index_list,"Brysbaert_Concreteness_Combined_FW_no_kw",header_list)
			
		#Subtlexus
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_freq,"all","0",index_list,"SUBTLEXus_Freq_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_cd,"all","0",index_list,"SUBTLEXus_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_freq_log,"all","0",index_list,"SUBTLEXus_Freq_AW_Log_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_cd_log,"all","0",index_list,"SUBTLEXus_Range_AW_Log_no_kw",header_list)

			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):

			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_freq,"cw","0",index_list,"SUBTLEXus_Freq_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_cd,"cw","0",index_list,"SUBTLEXus_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_freq_log,"cw","0",index_list,"SUBTLEXus_Freq_CW_Log_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_cd_log,"cw","0",index_list,"SUBTLEXus_Range_CW_Log_no_kw",header_list)
		
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
		
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_freq,"fw","0",index_list,"SUBTLEXus_Freq_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_cd,"fw","0",index_list,"SUBTLEXus_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_freq_log,"fw","0",index_list,"SUBTLEXus_Freq_FW_Log_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),subtlex_cd_log,"fw","0",index_list,"SUBTLEXus_Range_FW_Log_no_kw",header_list)
		
		#BNC_Word
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_freq_written,"all","0",index_list,"BNC_Written_Freq_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_freq_written_log,"all","0",index_list,"BNC_Written_Freq_AW_Log_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_range_written,"all","0",index_list,"BNC_Written_Range_AW_no_kw",header_list)
			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_freq_written,"cw","0",index_list,"BNC_Written_Freq_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_freq_written_log,"cw","0",index_list,"BNC_Written_Freq_CW_Log_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_range_written,"cw","0",index_list,"BNC_Written_Range_CW_no_kw",header_list)
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_freq_written,"fw","0",index_list,"BNC_Written_Freq_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_freq_written_log,"fw","0",index_list,"BNC_Written_Freq_FW_Log_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(clean_text,simple_source_keywords),bnc_range_written,"fw","0",index_list,"BNC_Written_Range_FW_no_kw",header_list)

		#COCA_academic_raw
			ktk.DataDict_counter(coca_text,COCA_academic_uni_R,"aw","0",index_list,"COCA_Academic_Range_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_F,"aw","0",index_list,"COCA_Academic_Frequency_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_R_log,"aw","0",index_list,"COCA_Academic_Range_Log_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_F_log,"aw","0",index_list,"COCA_Academic_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(coca_text,COCA_academic_uni_R,"cw","0",index_list,"COCA_Academic_Range_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_F,"cw","0",index_list,"COCA_Academic_Frequency_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_R_log,"cw","0",index_list,"COCA_Academic_Range_Log_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_F_log,"cw","0",index_list,"COCA_Academic_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(coca_text,COCA_academic_uni_R,"fw","0",index_list,"COCA_Academic_Range_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_F,"fw","0",index_list,"COCA_Academic_Frequency_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_R_log,"fw","0",index_list,"COCA_Academic_Range_Log_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_academic_uni_F_log,"fw","0",index_list,"COCA_Academic_Frequency_Log_FW",header_list)

		#COCA_fiction_raw
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_R,"aw","0",index_list,"COCA_fiction_Range_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_F,"aw","0",index_list,"COCA_fiction_Frequency_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_R_log,"aw","0",index_list,"COCA_fiction_Range_Log_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_F_log,"aw","0",index_list,"COCA_fiction_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_R,"cw","0",index_list,"COCA_fiction_Range_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_F,"cw","0",index_list,"COCA_fiction_Frequency_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_R_log,"cw","0",index_list,"COCA_fiction_Range_Log_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_F_log,"cw","0",index_list,"COCA_fiction_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(coca_text,COCA_fiction_uni_R,"fw","0",index_list,"COCA_fiction_Range_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_F,"fw","0",index_list,"COCA_fiction_Frequency_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_R_log,"fw","0",index_list,"COCA_fiction_Range_Log_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_fiction_uni_F_log,"fw","0",index_list,"COCA_fiction_Frequency_Log_FW",header_list)

		#COCA_magazine_raw
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_R,"aw","0",index_list,"COCA_magazine_Range_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_F,"aw","0",index_list,"COCA_magazine_Frequency_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_R_log,"aw","0",index_list,"COCA_magazine_Range_Log_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_F_log,"aw","0",index_list,"COCA_magazine_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_R,"cw","0",index_list,"COCA_magazine_Range_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_F,"cw","0",index_list,"COCA_magazine_Frequency_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_R_log,"cw","0",index_list,"COCA_magazine_Range_Log_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_F_log,"cw","0",index_list,"COCA_magazine_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(coca_text,COCA_magazine_uni_R,"fw","0",index_list,"COCA_magazine_Range_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_F,"fw","0",index_list,"COCA_magazine_Frequency_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_R_log,"fw","0",index_list,"COCA_magazine_Range_Log_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_magazine_uni_F_log,"fw","0",index_list,"COCA_magazine_Frequency_Log_FW",header_list)
		
		#COCA_newspaper_raw
			ktk.DataDict_counter(coca_text,COCA_news_uni_R,"aw","0",index_list,"COCA_news_Range_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_F,"aw","0",index_list,"COCA_news_Frequency_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_R_log,"aw","0",index_list,"COCA_news_Range_Log_AW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_F_log,"aw","0",index_list,"COCA_news_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(coca_text,COCA_news_uni_R,"cw","0",index_list,"COCA_news_Range_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_F,"cw","0",index_list,"COCA_news_Frequency_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_R_log,"cw","0",index_list,"COCA_news_Range_Log_CW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_F_log,"cw","0",index_list,"COCA_news_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(coca_text,COCA_news_uni_R,"fw","0",index_list,"COCA_news_Range_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_F,"fw","0",index_list,"COCA_news_Frequency_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_R_log,"fw","0",index_list,"COCA_news_Range_Log_FW",header_list)
			ktk.DataDict_counter(coca_text,COCA_news_uni_F_log,"fw","0",index_list,"COCA_news_Frequency_Log_FW",header_list)

		#COCA_academic
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_R,"aw","0",index_list,"COCA_Academic_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_F,"aw","0",index_list,"COCA_Academic_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_R_log,"aw","0",index_list,"COCA_Academic_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_F_log,"aw","0",index_list,"COCA_Academic_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_R,"cw","0",index_list,"COCA_Academic_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_F,"cw","0",index_list,"COCA_Academic_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_R_log,"cw","0",index_list,"COCA_Academic_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_F_log,"cw","0",index_list,"COCA_Academic_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_R,"fw","0",index_list,"COCA_Academic_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_F,"fw","0",index_list,"COCA_Academic_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_R_log,"fw","0",index_list,"COCA_Academic_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,acad_keywords),COCA_academic_uni_F_log,"fw","0",index_list,"COCA_Academic_Frequency_Log_FW_no_kw",header_list)

		#COCA_fiction
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_R,"aw","0",index_list,"COCA_fiction_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_F,"aw","0",index_list,"COCA_fiction_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_R_log,"aw","0",index_list,"COCA_fiction_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_F_log,"aw","0",index_list,"COCA_fiction_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_R,"cw","0",index_list,"COCA_fiction_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_F,"cw","0",index_list,"COCA_fiction_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_R_log,"cw","0",index_list,"COCA_fiction_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_F_log,"cw","0",index_list,"COCA_fiction_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_R,"fw","0",index_list,"COCA_fiction_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_F,"fw","0",index_list,"COCA_fiction_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_R_log,"fw","0",index_list,"COCA_fiction_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,fic_keywords),COCA_fiction_uni_F_log,"fw","0",index_list,"COCA_fiction_Frequency_Log_FW_no_kw",header_list)

		#COCA_magazine_raw
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_R,"aw","0",index_list,"COCA_magazine_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_F,"aw","0",index_list,"COCA_magazine_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_R_log,"aw","0",index_list,"COCA_magazine_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_F_log,"aw","0",index_list,"COCA_magazine_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_R,"cw","0",index_list,"COCA_magazine_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_F,"cw","0",index_list,"COCA_magazine_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_R_log,"cw","0",index_list,"COCA_magazine_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_F_log,"cw","0",index_list,"COCA_magazine_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_R,"fw","0",index_list,"COCA_magazine_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_F,"fw","0",index_list,"COCA_magazine_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_R_log,"fw","0",index_list,"COCA_magazine_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,mag_keywords),COCA_magazine_uni_F_log,"fw","0",index_list,"COCA_magazine_Frequency_Log_FW_no_kw",header_list)
		
		#COCA_newspaper_raw
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_R,"aw","0",index_list,"COCA_news_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_F,"aw","0",index_list,"COCA_news_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_R_log,"aw","0",index_list,"COCA_news_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_F_log,"aw","0",index_list,"COCA_news_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_R,"cw","0",index_list,"COCA_news_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_F,"cw","0",index_list,"COCA_news_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_R_log,"cw","0",index_list,"COCA_news_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_F_log,"cw","0",index_list,"COCA_news_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_R,"fw","0",index_list,"COCA_news_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_F,"fw","0",index_list,"COCA_news_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_R_log,"fw","0",index_list,"COCA_news_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(coca_text,news_keywords),COCA_news_uni_F_log,"fw","0",index_list,"COCA_news_Frequency_Log_FW_no_kw",header_list)

	##### Lemmatized Uni Frequency #####
		#COCA_lemma_acad_raw
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_AW",header_list)

			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_FW",header_list)

		#COCA_lemma_fic_raw
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_FW",header_list)

		#COCA_lemma_mag_raw
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_FW",header_list)
		
		#COCA_newspaper_raw
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_AW",header_list)
			ktk.DataDict_counter(pos_lem_dict["all"],COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_AW",header_list)
		
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_CW",header_list)
			ktk.DataDict_counter(pos_lem_dict["content"],COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_CW",header_list)

			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_FW",header_list)
			ktk.DataDict_counter(pos_lem_dict["function"],COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_FW",header_list)

		#COCA_academic
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_FW_no_kw",header_list)

		#COCA_fiction
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_FW_no_kw",header_list)

		#COCA_lemma_mag_raw
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_FW_no_kw",header_list)
		
		#COCA_newspaper_raw
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_AW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_AW_no_kw",header_list)
		
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_CW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_CW_no_kw",header_list)

			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_FW_no_kw",header_list)
			ktk.DataDict_counter(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_FW_no_kw",header_list)
	#### Lemmatized Uni Frequency ####
	
		#Psycholinguistic_Norms
			ktk.simple_sum(pos_dict["s_all"],fam,"all","0",index_list,"MRC_Familiarity_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],concreteness,"all","0",index_list,"MRC_Concreteness_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],imageability,"all","0",index_list,"MRC_Imageability_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],meaningfulness_colorado,"all","0",index_list,"MRC_Meaningfulness_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],fam,"cw","0",index_list,"MRC_Familiarity_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],concreteness,"cw","0",index_list,"MRC_Concreteness_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],imageability,"cw","0",index_list,"MRC_Imageability_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],fam,"fw","0",index_list,"MRC_Familiarity_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],concreteness,"fw","0",index_list,"MRC_Concreteness_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],imageability,"fw","0",index_list,"MRC_Imageability_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],meaningfulness_colorado,"fw","0",index_list,"MRC_Meaningfulness_FW_sntmin",header_list)
			
		#_Kuperman_et_al._AoA_Mean_Scores:
			ktk.simple_sum(pos_dict["s_all"],B_AoA,"all","0",index_list,"Kuperman_AoA_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],B_AoA,"cw","0",index_list,"Kuperman_AoA_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],B_AoA,"fw","0",index_list,"Kuperman_AoA_FW_sntmin",header_list)
			
		#Brysbaert_et_al_Unigram_Concreteness:
			ktk.simple_sum(pos_dict["s_all"],B_Conc,"aw",index_list,"Brysbaert_Concreteness_Uni_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],B_Conc,"cw",index_list,"Brysbaert_Concreteness_Uni_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],B_Conc,"fw",index_list,"Brysbaert_Concreteness_Uni_FW_sntmin",header_list)
			
		#Subtlexus
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.simple_sum(pos_dict["s_all"],subtlex_freq,"all","0",index_list,"SUBTLEXus_Freq_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_cd,"all","0",index_list,"SUBTLEXus_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_freq_log,"all","0",index_list,"SUBTLEXus_Freq_AW_Log_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_cd_log,"all","0",index_list,"SUBTLEXus_Range_AW_Log_sntmin",header_list)

			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):

			ktk.simple_sum(pos_dict["s_all"],subtlex_freq,"cw","0",index_list,"SUBTLEXus_Freq_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_cd,"cw","0",index_list,"SUBTLEXus_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_freq_log,"cw","0",index_list,"SUBTLEXus_Freq_CW_Log_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_cd_log,"cw","0",index_list,"SUBTLEXus_Range_CW_Log_sntmin",header_list)
		
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
		
			ktk.simple_sum(pos_dict["s_all"],subtlex_freq,"fw","0",index_list,"SUBTLEXus_Freq_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_cd,"fw","0",index_list,"SUBTLEXus_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_freq_log,"fw","0",index_list,"SUBTLEXus_Freq_FW_Log_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],subtlex_cd_log,"fw","0",index_list,"SUBTLEXus_Range_FW_Log_sntmin",header_list)
		
		#BNC_Word
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.simple_sum(pos_dict["s_all"],bnc_freq_written,"all","0",index_list,"BNC_Written_Freq_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],bnc_freq_written_log,"all","0",index_list,"BNC_Written_Freq_AW_Log_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],bnc_range_written,"all","0",index_list,"BNC_Written_Range_AW_sntmin",header_list)
			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):
			ktk.simple_sum(pos_dict["s_all"],bnc_freq_written,"cw","0",index_list,"BNC_Written_Freq_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],bnc_freq_written_log,"cw","0",index_list,"BNC_Written_Freq_CW_Log_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],bnc_range_written,"cw","0",index_list,"BNC_Written_Range_CW_sntmin",header_list)
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
			ktk.simple_sum(pos_dict["s_all"],bnc_freq_written,"fw","0",index_list,"BNC_Written_Freq_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],bnc_freq_written_log,"fw","0",index_list,"BNC_Written_Freq_FW_Log_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],bnc_range_written,"fw","0",index_list,"BNC_Written_Range_FW_sntmin",header_list)

		#Psycholinguistic_Norms
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),fam,"all","0",index_list,"MRC_Familiarity_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),concreteness,"all","0",index_list,"MRC_Concreteness_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),imageability,"all","0",index_list,"MRC_Imageability_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),meaningfulness_colorado,"all","0",index_list,"MRC_Meaningfulness_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),fam,"cw","0",index_list,"MRC_Familiarity_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),concreteness,"cw","0",index_list,"MRC_Concreteness_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),imageability,"cw","0",index_list,"MRC_Imageability_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),fam,"fw","0",index_list,"MRC_Familiarity_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),concreteness,"fw","0",index_list,"MRC_Concreteness_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),imageability,"fw","0",index_list,"MRC_Imageability_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),meaningfulness_colorado,"fw","0",index_list,"MRC_Meaningfulness_FW_no_kw_sntmin",header_list)
			
		#_Kuperman_et_al._AoA_Mean_Scores:
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),B_AoA,"all","0",index_list,"Kuperman_AoA_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),B_AoA,"cw","0",index_list,"Kuperman_AoA_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),B_AoA,"fw","0",index_list,"Kuperman_AoA_FW_no_kw_sntmin",header_list)
			
		#Brysbaert_et_al_Unigram_Concreteness:
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),B_Conc,"aw",index_list,"Brysbaert_Concreteness_unigrams_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),B_Conc,"cw",index_list,"Brysbaert_Concreteness_unigrams_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),B_Conc,"fw",index_list,"Brysbaert_Concreteness_unigrams_FW_no_kw_sntmin",header_list)
			
		#Subtlexus
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_freq,"all","0",index_list,"SUBTLEXus_Freq_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_cd,"all","0",index_list,"SUBTLEXus_Range_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_freq_log,"all","0",index_list,"SUBTLEXus_Freq_AW_Log_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_cd_log,"all","0",index_list,"SUBTLEXus_Range_AW_Log_no_kw_sntmin",header_list)

			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):

			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_freq,"cw","0",index_list,"SUBTLEXus_Freq_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_cd,"cw","0",index_list,"SUBTLEXus_Range_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_freq_log,"cw","0",index_list,"SUBTLEXus_Freq_CW_Log_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_cd_log,"cw","0",index_list,"SUBTLEXus_Range_CW_Log_no_kw_sntmin",header_list)
		
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
		
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_freq,"fw","0",index_list,"SUBTLEXus_Freq_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_cd,"fw","0",index_list,"SUBTLEXus_Range_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_freq_log,"fw","0",index_list,"SUBTLEXus_Freq_FW_Log_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),subtlex_cd_log,"fw","0",index_list,"SUBTLEXus_Range_FW_Log_no_kw_sntmin",header_list)
		
		#BNC_Word
			#Every_Word_Mean_Score_(Total_Index_Score/Number_of_Incidences,"0",,header_list):
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_freq_written,"all","0",index_list,"BNC_Written_Freq_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_freq_written_log,"all","0",index_list,"BNC_Written_Freq_AW_Log_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_range_written,"all","0",index_list,"BNC_Written_Range_AW_no_kw_sntmin",header_list)
			#Content_Word_Mean_Score_(Total_Content_Word_Index_Score/Number_of_Content_Words,"0",,header_list):
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_freq_written,"cw","0",index_list,"BNC_Written_Freq_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_freq_written_log,"cw","0",index_list,"BNC_Written_Freq_CW_Log_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_range_written,"cw","0",index_list,"BNC_Written_Range_CW_no_kw_sntmin",header_list)
			#Function_Word_Mean_Score_(Total_Function_Word_Index_Score/Number_of_Function_Words,"0",,header_list):
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_freq_written,"fw","0",index_list,"BNC_Written_Freq_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_freq_written_log,"fw","0",index_list,"BNC_Written_Freq_FW_Log_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],simple_source_keywords),bnc_range_written,"fw","0",index_list,"BNC_Written_Range_FW_no_kw_sntmin",header_list)

		#COCA_academic
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_R,"aw","0",index_list,"COCA_Academic_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_F,"aw","0",index_list,"COCA_Academic_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_R_log,"aw","0",index_list,"COCA_Academic_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_F_log,"aw","0",index_list,"COCA_Academic_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_R,"cw","0",index_list,"COCA_Academic_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_F,"cw","0",index_list,"COCA_Academic_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_R_log,"cw","0",index_list,"COCA_Academic_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_F_log,"cw","0",index_list,"COCA_Academic_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_R,"fw","0",index_list,"COCA_Academic_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_F,"fw","0",index_list,"COCA_Academic_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_R_log,"fw","0",index_list,"COCA_Academic_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_academic_uni_F_log,"fw","0",index_list,"COCA_Academic_Frequency_Log_FW_sntmin",header_list)

		#COCA_fiction
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_R,"aw","0",index_list,"COCA_fiction_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_F,"aw","0",index_list,"COCA_fiction_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_R_log,"aw","0",index_list,"COCA_fiction_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_F_log,"aw","0",index_list,"COCA_fiction_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_R,"cw","0",index_list,"COCA_fiction_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_F,"cw","0",index_list,"COCA_fiction_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_R_log,"cw","0",index_list,"COCA_fiction_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_F_log,"cw","0",index_list,"COCA_fiction_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_R,"fw","0",index_list,"COCA_fiction_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_F,"fw","0",index_list,"COCA_fiction_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_R_log,"fw","0",index_list,"COCA_fiction_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_fiction_uni_F_log,"fw","0",index_list,"COCA_fiction_Frequency_Log_FW_sntmin",header_list)

		#COCA_magazine
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_R,"aw","0",index_list,"COCA_magazine_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_F,"aw","0",index_list,"COCA_magazine_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_R_log,"aw","0",index_list,"COCA_magazine_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_F_log,"aw","0",index_list,"COCA_magazine_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_R,"cw","0",index_list,"COCA_magazine_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_F,"cw","0",index_list,"COCA_magazine_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_R_log,"cw","0",index_list,"COCA_magazine_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_F_log,"cw","0",index_list,"COCA_magazine_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_R,"fw","0",index_list,"COCA_magazine_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_F,"fw","0",index_list,"COCA_magazine_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_R_log,"fw","0",index_list,"COCA_magazine_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_magazine_uni_F_log,"fw","0",index_list,"COCA_magazine_Frequency_Log_FW_sntmin",header_list)
		
		#COCA_newspaper
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_R,"aw","0",index_list,"COCA_news_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_F,"aw","0",index_list,"COCA_news_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_R_log,"aw","0",index_list,"COCA_news_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_F_log,"aw","0",index_list,"COCA_news_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_R,"cw","0",index_list,"COCA_news_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_F,"cw","0",index_list,"COCA_news_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_R_log,"cw","0",index_list,"COCA_news_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_F_log,"cw","0",index_list,"COCA_news_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_R,"fw","0",index_list,"COCA_news_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_F,"fw","0",index_list,"COCA_news_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_R_log,"fw","0",index_list,"COCA_news_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_dict["s_all"],COCA_news_uni_F_log,"fw","0",index_list,"COCA_news_Frequency_Log_FW_sntmin",header_list)

		#COCA_academic
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_R,"aw","0",index_list,"COCA_Academic_Range_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_F,"aw","0",index_list,"COCA_Academic_Frequency_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_R_log,"aw","0",index_list,"COCA_Academic_Range_Log_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_F_log,"aw","0",index_list,"COCA_Academic_Frequency_Log_AW_no_kw_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_R,"cw","0",index_list,"COCA_Academic_Range_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_F,"cw","0",index_list,"COCA_Academic_Frequency_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_R_log,"cw","0",index_list,"COCA_Academic_Range_Log_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_F_log,"cw","0",index_list,"COCA_Academic_Frequency_Log_CW_no_kw_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_R,"fw","0",index_list,"COCA_Academic_Range_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_F,"fw","0",index_list,"COCA_Academic_Frequency_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_R_log,"fw","0",index_list,"COCA_Academic_Range_Log_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],acad_keywords),COCA_academic_uni_F_log,"fw","0",index_list,"COCA_Academic_Frequency_Log_FW_no_kw_sntmin",header_list)

		#COCA_fiction
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_R,"aw","0",index_list,"COCA_fiction_Range_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_F,"aw","0",index_list,"COCA_fiction_Frequency_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_R_log,"aw","0",index_list,"COCA_fiction_Range_Log_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_F_log,"aw","0",index_list,"COCA_fiction_Frequency_Log_AW_no_kw_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_R,"cw","0",index_list,"COCA_fiction_Range_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_F,"cw","0",index_list,"COCA_fiction_Frequency_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_R_log,"cw","0",index_list,"COCA_fiction_Range_Log_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_F_log,"cw","0",index_list,"COCA_fiction_Frequency_Log_CW_no_kw_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_R,"fw","0",index_list,"COCA_fiction_Range_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_F,"fw","0",index_list,"COCA_fiction_Frequency_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_R_log,"fw","0",index_list,"COCA_fiction_Range_Log_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],fic_keywords),COCA_fiction_uni_F_log,"fw","0",index_list,"COCA_fiction_Frequency_Log_FW_no_kw_sntmin",header_list)

		#COCA_magazine
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_R,"aw","0",index_list,"COCA_magazine_Range_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_F,"aw","0",index_list,"COCA_magazine_Frequency_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_R_log,"aw","0",index_list,"COCA_magazine_Range_Log_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_F_log,"aw","0",index_list,"COCA_magazine_Frequency_Log_AW_no_kw_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_R,"cw","0",index_list,"COCA_magazine_Range_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_F,"cw","0",index_list,"COCA_magazine_Frequency_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_R_log,"cw","0",index_list,"COCA_magazine_Range_Log_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_F_log,"cw","0",index_list,"COCA_magazine_Frequency_Log_CW_no_kw_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_R,"fw","0",index_list,"COCA_magazine_Range_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_F,"fw","0",index_list,"COCA_magazine_Frequency_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_R_log,"fw","0",index_list,"COCA_magazine_Range_Log_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],mag_keywords),COCA_magazine_uni_F_log,"fw","0",index_list,"COCA_magazine_Frequency_Log_FW_no_kw_sntmin",header_list)
		
		#COCA_newspaper
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_R,"aw","0",index_list,"COCA_news_Range_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_F,"aw","0",index_list,"COCA_news_Frequency_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_R_log,"aw","0",index_list,"COCA_news_Range_Log_AW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_F_log,"aw","0",index_list,"COCA_news_Frequency_Log_AW_no_kw_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_R,"cw","0",index_list,"COCA_news_Range_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_F,"cw","0",index_list,"COCA_news_Frequency_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_R_log,"cw","0",index_list,"COCA_news_Range_Log_CW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_F_log,"cw","0",index_list,"COCA_news_Frequency_Log_CW_no_kw_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_R,"fw","0",index_list,"COCA_news_Range_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_F,"fw","0",index_list,"COCA_news_Frequency_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_R_log,"fw","0",index_list,"COCA_news_Range_Log_FW_no_kw_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_dict["s_all"],news_keywords),COCA_news_uni_F_log,"fw","0",index_list,"COCA_news_Frequency_Log_FW_no_kw_sntmin",header_list)

	#### Lemmatized Uni Frequency Sentence Minimum ####
			#COCA_lemma_acad_raw
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_AW_sntmin",header_list)

			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_FW_sntmin",header_list)

		#COCA_lemma_fic_raw
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_FW_sntmin",header_list)

		#COCA_lemma_mag_raw
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_FW_sntmin",header_list)
		
		#COCA_newspaper_raw
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_AW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["all"],COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_AW_sntmin",header_list)
		
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_CW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["content"],COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_CW_sntmin",header_list)

			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_FW_sntmin",header_list)
			ktk.simple_sum(pos_lem_dict["function"],COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_FW_sntmin",header_list)

		#COCA_academic
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],acad_keywords),COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_AW_no_kW_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],acad_keywords),COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_CW_no_kW_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_R,"aw","0",index_list,"COCA_lemma_acad_Range_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_F,"aw","0",index_list,"COCA_lemma_acad_Frequency_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_R_log,"aw","0",index_list,"COCA_lemma_acad_Range_Log_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],acad_keywords),COCA_lemma_acad_uni_F_log,"aw","0",index_list,"COCA_lemma_acad_Frequency_Log_FW_no_kW_sntmin",header_list)

		#COCA_fiction
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],fic_keywords),COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_AW_no_kW_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],fic_keywords),COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_CW_no_kW_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_R,"aw","0",index_list,"COCA_lemma_fic_Range_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_F,"aw","0",index_list,"COCA_lemma_fic_Frequency_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_R_log,"aw","0",index_list,"COCA_lemma_fic_Range_Log_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],fic_keywords),COCA_lemma_fic_uni_F_log,"aw","0",index_list,"COCA_lemma_fic_Frequency_Log_FW_no_kW_sntmin",header_list)

		#COCA_lemma_mag_raw
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],mag_keywords),COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_AW_no_kW_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],mag_keywords),COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_CW_no_kW_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_R,"aw","0",index_list,"COCA_lemma_mag_Range_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_F,"aw","0",index_list,"COCA_lemma_mag_Frequency_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_R_log,"aw","0",index_list,"COCA_lemma_mag_Range_Log_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],mag_keywords),COCA_lemma_mag_uni_F_log,"aw","0",index_list,"COCA_lemma_mag_Frequency_Log_FW_no_kW_sntmin",header_list)
		
		#COCA_newspaper_raw
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_AW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["all"],news_keywords),COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_AW_no_kW_sntmin",header_list)
		
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_CW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["content"],news_keywords),COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_CW_no_kW_sntmin",header_list)

			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_R,"aw","0",index_list,"COCA_lemma_news_Range_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_F,"aw","0",index_list,"COCA_lemma_news_Frequency_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_R_log,"aw","0",index_list,"COCA_lemma_news_Range_Log_FW_no_kW_sntmin",header_list)
			ktk.simple_sum(ktk.constrainer(pos_lem_dict["function"],news_keywords),COCA_lemma_news_uni_F_log,"aw","0",index_list,"COCA_lemma_news_Frequency_Log_FW_no_kW_sntmin",header_list)
	#### Lemmatized Uni Frequency Sentence Minimum ####
	
		#lsa_similarity
			#all 
			ktk.lsa_similarity(source_pos_dict["all"],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_all_rwd",header_list,"rwd")
			ktk.lsa_similarity(source_pos_dict["all"],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_all_fwd",header_list,"fwd")
			ktk.lsa_similarity(source_pos_dict["all"],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_all_normal",header_list,"normal")

			
			source_length = len(source_pos_dict["all"])
			one_third = int(source_length/3)
			two_thirds = one_third + one_third
			#first third source
			ktk.lsa_similarity(source_pos_dict["all"][:one_third],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_1-3_rwd",header_list,"rwd")
			ktk.lsa_similarity(source_pos_dict["all"][:one_third],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_1-3_fwd",header_list,"fwd")
			ktk.lsa_similarity(source_pos_dict["all"][:one_third],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_1-3_normal",header_list,"normal")

			#second third source
			ktk.lsa_similarity(source_pos_dict["all"][one_third:two_thirds],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_2-3_rwd",header_list,"rwd")
			ktk.lsa_similarity(source_pos_dict["all"][one_third:two_thirds],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_2-3_fwd",header_list,"fwd")
			ktk.lsa_similarity(source_pos_dict["all"][one_third:two_thirds],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_2-3_normal",header_list,"normal")
			
			#final third source
			ktk.lsa_similarity(source_pos_dict["all"][two_thirds:],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_3-3_rwd",header_list,"rwd")
			ktk.lsa_similarity(source_pos_dict["all"][two_thirds:],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_3-3_fwd",header_list,"fwd")
			ktk.lsa_similarity(source_pos_dict["all"][two_thirds:],pos_dict["all"],lsa_matrix,lsa_weights,index_list,"lsa_3-3_normal",header_list,"normal")
		
		#wordnet similarity
			ktk.syn_overlap(pos_dict["noun"], source_pos_dict["noun"], wn_noun_dict,index_list,"syn_overlap_nouns",header_list)
			ktk.syn_overlap(pos_dict["verb"], source_pos_dict["verb"], wn_verb_dict,index_list,"syn_overlap_verbs",header_list)
			
			noun_syn_overlap = ktk.syn_overlap(pos_dict["noun"], source_pos_dict["noun"], wn_noun_dict, list = "yes")
			verb_syn_overlap = ktk.syn_overlap(pos_dict["verb"], source_pos_dict["verb"], wn_verb_dict, list = "yes")
			index_list.append(ktk.safe_divide((noun_syn_overlap[0]+verb_syn_overlap[0]),(noun_syn_overlap[1]+verb_syn_overlap[1])))
			header_list.append("syn_overlap_verbs_nouns")
			
						
		#keyword similarity percentage
			#this is the model... will update when new lists are available
			#ktk.simple_proportion(clean_text,simple_source_keywords,"perc","simple_keywords_percentage",index_list,header_list)
			
			#acad
			ktk.simple_proportion(pos_lem_dict["all"],acad_keywords, "perc", "acad_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],acad_n_keywords, "perc", "acad_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],acad_no_pn_nkeywords, "perc", "acad_no_pn_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],acad_pn_keywords, "perc", "acad_pn_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],acad_v_keywords, "perc", "acad_v_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],acad_v_n_keywords, "perc", "acad_v_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],acad_adj_keywords, "perc", "acad_adj_uni_keywords_percentage", index_list, header_list)
			
			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],acad_bi_keywords, "perc", "acad_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],acad_tri_keywords, "perc", "acad_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],acad_quad_keywords, "perc", "acad_quad_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],acad_n_bi_keywords, "perc", "acad_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],acad_adj_bi_keywords, "perc", "acad_adj_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],acad_v_bi_keywords, "perc", "acad_v_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],acad_v_n_bi_keywords, "perc", "acad_v_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],acad_a_n_bi_keywords, "perc", "acad_a_n_bi_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],acad_n_tri_keywords, "perc", "acad_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],acad_adj_tri_keywords, "perc", "acad_adj_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],acad_v_tri_keywords, "perc", "acad_v_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],acad_v_n_tri_keywords, "perc", "acad_v_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],acad_a_n_tri_keywords, "perc", "acad_a_n_tri_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],acad_n_quad_keywords, "perc", "acad_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],acad_adj_quad_keywords, "perc", "acad_adj_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],acad_v_quad_keywords, "perc", "acad_v_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],acad_v_n_quad_keywords, "perc", "acad_v_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],acad_a_n_quad_keywords, "perc", "acad_a_n_quad_keywords_percentage", index_list, header_list)
			
			#fiction
			ktk.simple_proportion(pos_lem_dict["all"],fic_keywords, "perc", "fic_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],fic_n_keywords, "perc", "fic_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],fic_no_pn_nkeywords, "perc", "fic_no_pn_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],fic_pn_keywords, "perc", "fic_pn_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],fic_v_keywords, "perc", "fic_v_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],fic_v_n_keywords, "perc", "fic_v_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],fic_adj_keywords, "perc", "fic_adj_uni_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],fic_bi_keywords, "perc", "fic_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],fic_tri_keywords, "perc", "fic_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],fic_quad_keywords, "perc", "fic_quad_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],fic_n_bi_keywords, "perc", "fic_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],fic_adj_bi_keywords, "perc", "fic_adj_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],fic_v_bi_keywords, "perc", "fic_v_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],fic_v_n_bi_keywords, "perc", "fic_v_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],fic_a_n_bi_keywords, "perc", "fic_a_n_bi_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],fic_n_tri_keywords, "perc", "fic_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],fic_adj_tri_keywords, "perc", "fic_adj_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],fic_v_tri_keywords, "perc", "fic_v_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],fic_v_n_tri_keywords, "perc", "fic_v_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],fic_a_n_tri_keywords, "perc", "fic_a_n_tri_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],fic_n_quad_keywords, "perc", "fic_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],fic_adj_quad_keywords, "perc", "fic_adj_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],fic_v_quad_keywords, "perc", "fic_v_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],fic_v_n_quad_keywords, "perc", "fic_v_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],fic_a_n_quad_keywords, "perc", "fic_a_n_quad_keywords_percentage", index_list, header_list)

			#magazine
			
			ktk.simple_proportion(pos_lem_dict["all"],mag_keywords, "perc", "mag_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],mag_n_keywords, "perc", "mag_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],mag_no_pn_nkeywords, "perc", "mag_no_pn_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],mag_pn_keywords, "perc", "mag_pn_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],mag_v_keywords, "perc", "mag_v_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],mag_v_n_keywords, "perc", "mag_v_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],mag_adj_keywords, "perc", "mag_adj_uni_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],mag_bi_keywords, "perc", "mag_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],mag_tri_keywords, "perc", "mag_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],mag_quad_keywords, "perc", "mag_quad_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],mag_n_bi_keywords, "perc", "mag_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],mag_adj_bi_keywords, "perc", "mag_adj_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],mag_v_bi_keywords, "perc", "mag_v_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],mag_v_n_bi_keywords, "perc", "mag_v_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],mag_a_n_bi_keywords, "perc", "mag_a_n_bi_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],mag_n_tri_keywords, "perc", "mag_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],mag_adj_tri_keywords, "perc", "mag_adj_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],mag_v_tri_keywords, "perc", "mag_v_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],mag_v_n_tri_keywords, "perc", "mag_v_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],mag_a_n_tri_keywords, "perc", "mag_a_n_tri_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],mag_n_quad_keywords, "perc", "mag_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],mag_adj_quad_keywords, "perc", "mag_adj_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],mag_v_quad_keywords, "perc", "mag_v_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],mag_v_n_quad_keywords, "perc", "mag_v_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],mag_a_n_quad_keywords, "perc", "mag_a_n_quad_keywords_percentage", index_list, header_list)

			#news
			ktk.simple_proportion(pos_lem_dict["all"],news_keywords, "perc", "news_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],news_n_keywords, "perc", "news_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],news_no_pn_nkeywords, "perc", "news_no_pn_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],news_pn_keywords, "perc", "news_pn_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],news_v_keywords, "perc", "news_v_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],news_v_n_keywords, "perc", "news_v_n_uni_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],news_adj_keywords, "perc", "news_adj_uni_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],news_bi_keywords, "perc", "news_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],news_tri_keywords, "perc", "news_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],news_quad_keywords, "perc", "news_quad_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],news_n_bi_keywords, "perc", "news_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],news_adj_bi_keywords, "perc", "news_adj_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],news_v_bi_keywords, "perc", "news_v_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],news_v_n_bi_keywords, "perc", "news_v_n_bi_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],news_a_n_bi_keywords, "perc", "news_a_n_bi_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],news_n_tri_keywords, "perc", "news_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],news_adj_tri_keywords, "perc", "news_adj_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],news_v_tri_keywords, "perc", "news_v_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],news_v_n_tri_keywords, "perc", "news_v_n_tri_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],news_a_n_tri_keywords, "perc", "news_a_n_tri_keywords_percentage", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],news_n_quad_keywords, "perc", "news_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],news_adj_quad_keywords, "perc", "news_adj_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],news_v_quad_keywords, "perc", "news_v_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],news_v_n_quad_keywords, "perc", "news_v_n_quad_keywords_percentage", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],news_a_n_quad_keywords, "perc", "news_a_n_quad_keywords_percentage", index_list, header_list)

		#keyword similarity proportion
			#this is the model... will update when new lists are available
			#ktk.simple_proportion(clean_text,simple_source_keywords,"prop","simple_keywords_proportion",index_list,header_list)

			#acad
			ktk.simple_proportion(pos_lem_dict["all"],acad_keywords, "prop", "acad_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],acad_n_keywords, "prop", "acad_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],acad_no_pn_nkeywords, "prop", "acad_no_pn_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],acad_pn_keywords, "prop", "acad_pn_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],acad_v_keywords, "prop", "acad_v_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],acad_v_n_keywords, "prop", "acad_v_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],acad_adj_keywords, "prop", "acad_adj_uni_keywords_proportion", index_list, header_list)
			
			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],acad_bi_keywords, "prop", "acad_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],acad_tri_keywords, "prop", "acad_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],acad_quad_keywords, "prop", "acad_quad_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],acad_n_bi_keywords, "prop", "acad_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],acad_adj_bi_keywords, "prop", "acad_adj_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],acad_v_bi_keywords, "prop", "acad_v_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],acad_v_n_bi_keywords, "prop", "acad_v_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],acad_a_n_bi_keywords, "prop", "acad_a_n_bi_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],acad_n_tri_keywords, "prop", "acad_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],acad_adj_tri_keywords, "prop", "acad_adj_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],acad_v_tri_keywords, "prop", "acad_v_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],acad_v_n_tri_keywords, "prop", "acad_v_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],acad_a_n_tri_keywords, "prop", "acad_a_n_tri_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],acad_n_quad_keywords, "prop", "acad_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],acad_adj_quad_keywords, "prop", "acad_adj_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],acad_v_quad_keywords, "prop", "acad_v_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],acad_v_n_quad_keywords, "prop", "acad_v_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],acad_a_n_quad_keywords, "prop", "acad_a_n_quad_keywords_proportion", index_list, header_list)
			
			#fiction
			ktk.simple_proportion(pos_lem_dict["all"],fic_keywords, "prop", "fic_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],fic_n_keywords, "prop", "fic_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],fic_no_pn_nkeywords, "prop", "fic_no_pn_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],fic_pn_keywords, "prop", "fic_pn_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],fic_v_keywords, "prop", "fic_v_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],fic_v_n_keywords, "prop", "fic_v_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],fic_adj_keywords, "prop", "fic_adj_uni_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],fic_bi_keywords, "prop", "fic_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],fic_tri_keywords, "prop", "fic_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],fic_quad_keywords, "prop", "fic_quad_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],fic_n_bi_keywords, "prop", "fic_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],fic_adj_bi_keywords, "prop", "fic_adj_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],fic_v_bi_keywords, "prop", "fic_v_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],fic_v_n_bi_keywords, "prop", "fic_v_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],fic_a_n_bi_keywords, "prop", "fic_a_n_bi_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],fic_n_tri_keywords, "prop", "fic_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],fic_adj_tri_keywords, "prop", "fic_adj_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],fic_v_tri_keywords, "prop", "fic_v_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],fic_v_n_tri_keywords, "prop", "fic_v_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],fic_a_n_tri_keywords, "prop", "fic_a_n_tri_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],fic_n_quad_keywords, "prop", "fic_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],fic_adj_quad_keywords, "prop", "fic_adj_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],fic_v_quad_keywords, "prop", "fic_v_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],fic_v_n_quad_keywords, "prop", "fic_v_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],fic_a_n_quad_keywords, "prop", "fic_a_n_quad_keywords_proportion", index_list, header_list)

			#magazine
			ktk.simple_proportion(pos_lem_dict["all"],mag_keywords, "prop", "mag_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],mag_n_keywords, "prop", "mag_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],mag_no_pn_nkeywords, "prop", "mag_no_pn_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],mag_pn_keywords, "prop", "mag_pn_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],mag_v_keywords, "prop", "mag_v_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],mag_v_n_keywords, "prop", "mag_v_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],mag_adj_keywords, "prop", "mag_adj_uni_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],mag_bi_keywords, "prop", "mag_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],mag_tri_keywords, "prop", "mag_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],mag_quad_keywords, "prop", "mag_quad_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],mag_n_bi_keywords, "prop", "mag_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],mag_adj_bi_keywords, "prop", "mag_adj_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],mag_v_bi_keywords, "prop", "mag_v_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],mag_v_n_bi_keywords, "prop", "mag_v_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],mag_a_n_bi_keywords, "prop", "mag_a_n_bi_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],mag_n_tri_keywords, "prop", "mag_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],mag_adj_tri_keywords, "prop", "mag_adj_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],mag_v_tri_keywords, "prop", "mag_v_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],mag_v_n_tri_keywords, "prop", "mag_v_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],mag_a_n_tri_keywords, "prop", "mag_a_n_tri_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],mag_n_quad_keywords, "prop", "mag_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],mag_adj_quad_keywords, "prop", "mag_adj_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],mag_v_quad_keywords, "prop", "mag_v_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],mag_v_n_quad_keywords, "prop", "mag_v_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],mag_a_n_quad_keywords, "prop", "mag_a_n_quad_keywords_proportion", index_list, header_list)

			#news
			ktk.simple_proportion(pos_lem_dict["all"],news_keywords, "prop", "news_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["noun"],news_n_keywords, "prop", "news_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["no_proper"],news_no_pn_nkeywords, "prop", "news_no_pn_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["proper_n"],news_pn_keywords, "prop", "news_pn_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb"],news_v_keywords, "prop", "news_v_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["verb_noun"],news_v_n_keywords, "prop", "news_v_n_uni_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(pos_lem_dict["adj"],news_adj_keywords, "prop", "news_adj_uni_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["bi_list"],news_bi_keywords, "prop", "news_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["tri_list"],news_tri_keywords, "prop", "news_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["quad_list"],news_quad_keywords, "prop", "news_quad_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_bi"],news_n_bi_keywords, "prop", "news_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_bi"],news_adj_bi_keywords, "prop", "news_adj_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_bi"],news_v_bi_keywords, "prop", "news_v_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_bi"],news_v_n_bi_keywords, "prop", "news_v_n_bi_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_bi"],news_a_n_bi_keywords, "prop", "news_a_n_bi_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_tri"],news_n_tri_keywords, "prop", "news_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_tri"],news_adj_tri_keywords, "prop", "news_adj_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_tri"],news_n_tri_keywords, "prop", "news_v_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_tri"],news_v_n_tri_keywords, "prop", "news_v_n_tri_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_tri"],news_a_n_tri_keywords, "prop", "news_a_n_tri_keywords_proportion", index_list, header_list)

			ktk.simple_proportion(ngram_pos_lem_dict["n_list_quad"],news_n_quad_keywords, "prop", "news_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["adj_list_quad"],news_adj_quad_keywords, "prop", "news_adj_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_list_quad"],news_v_quad_keywords, "prop", "news_v_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["v_n_list_quad"],news_v_n_quad_keywords, "prop", "news_v_n_quad_keywords_proportion", index_list, header_list)
			ktk.simple_proportion(ngram_pos_lem_dict["a_n_list_quad"],news_a_n_quad_keywords, "prop", "news_a_n_quad_keywords_proportion", index_list, header_list)

		###write phase of program ###	
			index_string_list=[] 
			
			if file_number == 0:
				#print "header string should print"
				header_string = ",".join(header_list)
				outf.write ('{0}\n'
				.format(header_string))
					
			for items in index_list:
				index_string_list.append(str(items))
			string = ",".join(index_string_list)
	
			outf.write ('{0},{1}\n'
			.format(filename.split("/")[-1],string))
			
			file_number+=1
			
		outf.flush()#flushes out buffer to clean output file
		outf.close()#close output file	
		finishmessage = ("Processed " + str(nfiles) + " Files")
		dataQueue.put(finishmessage)
		root.update_idletasks()
		#self.progress.config(text =finishmessage)
		tkMessageBox.showinfo("Finished!", "Your files have been processed by SMART!")

		
class Catcher:#This class watches and waits for an error message
	def __init__(self, func, subst, widget):
		self.func = func
		self.subst = subst
		self.widget = widget

	def __call__(self, *args):
		try:
			if self.subst:
				args = apply(self.subst, args)
			return apply(self.func, args)
		except SystemExit, msg:
			raise SystemExit, msg
		except:
			import traceback
			import tkMessageBox
			ermessage = traceback.format_exc(1)
			ermessage = re.sub(r'.*(?=Error)', "", ermessage, flags=re.DOTALL)
			ermessage = "There was a problem processing your files:\n\n"+ermessage
			tkMessageBox.showerror("Error Message", ermessage)

if __name__ == '__main__':		
	root = tk.Tk()
	root.wm_title("CRAT 1.0")
	root.configure(background = color)
	#sets starting size: NOTE: it doesn't appear as though Tkinter will let you make the 
	#starting size smaller than the space needed for widgets.
	root.geometry(geom_size)
	tk.CallWrapper = Catcher
	myapp = MyApp(root)
	root.mainloop()

