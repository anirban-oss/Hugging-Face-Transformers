from transformers import pipeline

model = pipeline(task="question-answering", model="deepset/roberta-base-squad2")

# context = """
# The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy mainly as light, ultraviolet, and infrared radiation. It is the most important source of energy for life on Earth.

# The Sun's diameter is about 1.39 million kilometers (864,000 miles), or 109 times that of Earth. Its mass is about 330,000 times that of Earth, comprising about 99.86% of the total mass of the Solar System. Roughly three-quarters of the Sun's mass consists of hydrogen (~73%); the rest is mostly helium (~25%), with much smaller quantities of heavier elements, including oxygen, carbon, neon, and iron.
# """
context = """
Anirban Chatterjee
Address: Mirpur, Kharagpur PIN – 721301
District – West Midnapore, West Bengal, India
E-mail: anirbanc88@gmail.com
Phone number: +91 9641091557
Education:
 Alliance Française du Bengale, Kolkata, India
(2019)
A1 (a1.1+ a1.2) level course in French language
Appreciation: 94.44 in A1 level
 Attended Jadavpur University, Kolkata, India (2016-2018)
Bachelor of Arts (honours) in philosophy

Attended Netaji Subhash Engineering College, Kolkata, India (2015-
2016)
Bachelor of Technology in Bio-medical engineering

Senior School Certificate Examination (2015)
Conducted by: Central Board of Secondary Education, Delhi, India
School: Kendriya Vidyalaya No.2, Kharagpur, India
Subjects: English, Physics, Mathematics, Chemistry, Biology
Percentage obtained (best of 4 subjects): 76.25%
Special mention: 95/100 in English language

Secondary School Examination (Session: 2011-2013)
Conducted by: Central Board of Secondary Education, Delhi, India
School: Kendriya Vidyalaya No.2, Kharagpur, India
Subjects: English, Hindi, Mathematics, Science, Social Science
Percentage obtained: 95% (10/10 CGPA)
Special mention: 10/10 Grade point in all subjects
Training in Music:
 Dotara (2016-2018): taught by Subhabrata Sen
Dotara is a two, four or five stringed folk instrument, originating from Indian
subcontinent, resembling a sarod.
 Khamak (2016-2018): taught by Subhabrata Sen
Khamak is a stringed percussion instrument, originating in India, close to ektara.
 Guitar (2017): self-taught from online courses by James Devon
 Studied musical ethnology in folk (baul) tradition of Bengal
 Tabla (2009-2011): Pracheen Kala Kendra, Chandigarh, India.
Awarded first divisionTraining in Art:
 Fine arts (2003-2007): Rabindra Charukala Parishad, Kolkata, India.
Awarded first division & gold medal
Training in Photography:
 Basic training session in photography (2016): by Jadavpur University
Photographic Club, Kolkata
Workshops in India:
 Chennai photo Biennale: March 2019
Attended lectures and film screenings, visited various photo galleries and
participated in “Namma Chennai Namma Biennale” Photography Contest
 Kochi Muziris Biennale: February 2019
Attended Amelia Rauser Art History workshop
 Indian film project 24 hour challenge, Spring Fest (IIT Kgp): January 2019
Participated in workshop & competition organised by Indian Institute of
Technology, Kharagpur
 Calcutta school of Music workshop: January 2019
Attended 1) Lecture on Indian classical music – Demonstration on Indian ragas by
Pandit Suman Ghosh
2) Western Classical Music Appreciation Workshop: The Art of Listening by
Santanu Datta
 Maya Art Space, Kolkata: March 2018
Successfully completed course on appreciation of contemporary art and visual
culture
 Internship - Missing Girls (Art and technology anti-trafficking
campaign.):
February-March 2018
 Summer Science Camp - University of Burdwan, India: June 2013
Certificate of participation for Inspire Internship 2013 program organised by
Department of Environmental Science, University of Burdwan
 Introduction to Computer Programming - IIT Kharagpur, India: May
2012
Attended summer school on introduction to programming through Python and C
language conducted by Department of Computer Science and Engineering, Indian
Institute of Technology, Kharagpur
Seminars in India:
 “Of calibrations and conversations”, or how to measure time in
walking: February 2019 - By Samprati Pani & Sarnath Banerjee, in
conversation with Sarover Zaidi, Centre for the Study of Developing Societies, New
Delhi
 Florentine Renaissance: October 2018
Lecture on History & Art by Urvi Mukhopadhyay Neuro-Cognitive Aspects of Music at Sir C.V. Raman Centre for Physics
and music, Jadavpur University: December 2016
 Acharya Prafulla Chandra Ray Chemistry seminar, IIT Kharagpur:
August 2014
Online open coursewares:
 Sustainable Cities, SDG Academy
 Human behavioural biology, Stanford university by Prof. Robert
Sapolsky
 Introduction to psychology, Yale University by Prof. Paul Bloom
 Physics: classical mechanics, Massachusetts Institute of Technology by
Prof. Walter Lewin
 Listening to music: music theory, Yale University by Prof. Craig Wright
 Contemporary India, University of Melbourne by Prof. Anthony D’Costa
 Justice, Harvard University by Prof. Michael Sandel
Courseworks attended in College:
 Jadavpur university:
Comparative cultural studies, English literature & culture, Sociology, Micro-
Economics, Macro-Economics, Linguistics, Indian Philosophy, Western Philosophy,
Indian logic, Western logic, Indian Ethics, Western Ethics, Psychology, Indian
Metaphysics, Western Metaphysics, Indian Epistemology, Western Epistemology,
Literature & photography, Literature & painting, Literature & theatre, Literature &
music, Literature & film, Comparative Literature, Introduction to psychoanalysis
 Engineering-science coursework:
Basic Engineering Science (Physics, Chemistry, Mathematics), Engineering Drawing
Workshop using Lathe Machine
Scholarships:
 Adamas University (2015):
Obtained Certificate of merit and 100% scholarship by ranking 3rd in Adamas
university scholarship examination in association with Anandabazar Patrika.
 Awarded 5000 rupees scholarship for outstanding performance in
Secondary School examination (2013) with CGPA 10
 Pathfinder talent search exam (2012):
Certificate of Merit for qualifying in stage 1 and 2 of PTSE junior successfully and
got scholarship to study there
National Examinations & Competitions:
 Qualified in both levels of Railway Recruitment Board exam: 2018-19 National Genius Search Examination conducted by National Genius
Search Foundation: 2013
Certificate of excellence for scoring 91.09 percentile in national genius search
examination

International Olympiad of science organised by Silverzone foundation:
Ranked 1st in class from 2011, 2012, 2013
Special Mention: Ranked within top 10 in state level from 2012, 2013.
 National Science Olympiad organised by Science Olympiad Foundation:
Ranked 1st in class from 2009,2010,2011,2012
Special Mention: Ranked within top 100 in All India Rank with percentile score of 99
in 2010, 2012

International Olympiad of mathematics organised by Silverzone
Foundation:
Ranked within top 5 in class from 2010, 2011, 2012 and 2013
Special Mention: Ranked in top 100 in state level from 2012, 2013
 International mathematics Olympiad organised by Science Olympiad
Foundation:
Ranked within top 5 in class from 2009, 2010, 2011, 2012 and 2013
Special Mention: Ranked within top 50 in state level in 2013
 Mathematics talent search examination: 2009
Secured 2ndrank in school level
 National cyber Olympiad (computer science) organised by Science
Olympiad Foundation: Ranked 1st in class from 2007, 2009 and 2011.
 Participated in GREEN OLYMPIAD (environmental science) 2011
conducted by Ministry of Environment and Forest, Government of India.
 National Level Science Talent Search Examination conducted by
Unified Council, India:
Ranked in top 200 in state level from 2010, 2011 and 2012
Special Mention - Ranked within top 50 in state level from 2013
Participated in International English Olympiad 2014 organised by
Science Olympiad Foundation and ranked 94 in state (Top 100)
 WIZ National Spell bee:

Secured 9th rank in west Bengal state level grand finale of wiz national spell bee
2012-13 held at South city international school, Kolkata
 Secured 3rd position in spelling contest 2009 held at Kendriya
Vidyalaya no.2, Kharagpur.
Quiz Competitions:
 Secured 2nd position in National Science Day 2014 quiz held at
Kendriya Vidyalaya no.2, Kharagpur.
 KVS National Social science Quiz
Cluster level social science exhibition 2013: 2 ndposition in quiz
 KVS National Science Quiz
Regional science exhibition 2013: 2nd position in quiz and our model
(insect catcher) got selected for national level exhibition. Secured 3rd position in National Mathematics Year 2012 Quiz held at
Kendriya Vidyalaya no.2, Kharagpur.
 Secured 2nd position in Annual School Quiz competition, 2012
 Secured 1st position in Annual School Quiz competition, 2009
 Certificate of participation in finals of Maggie good food quiz, 2008
 Kendriya Vidyalaya Sangathan foundation day quiz, 2007: 2 nd place in
junior group
Social Work:
 Helpage India:
Certificate of social service for creating awareness and assistance in raising funds
for the care of elderly, irrespective of race, religion, caste or creed.
 CPAIDS cancer patient’s aid society:
Certificate of enforcement - participated as volunteer in mass
campaign/fight/awareness against cancer eradication
Blood Donation:
 Certificate of appreciation by state blood transfusion council, West
Bengal:
For donating blood voluntarily at voluntary blood donation camp organised at
Avijatri club, Kharagpur in association with state government hospital blood bank.
2019

Certificate of appreciation by state blood transfusion council, West
Bengal:
For donating blood voluntarily at voluntary blood donation camp organised at
Jadavpur University, Kolkata-32 in association with state government hospital
blood bank. 2016
Bharat Scouts and guides:
 Certificate of participation in the scouts and guides camp and
successfully passed “Pratham sopan” held in Kendriya Vidyalaya No.2
Kharagpur from 18/10 – 19/10/08
 Certificate of participation in the cubs and bulbuls camp and
successfully passed: “Pravesh” held in Kendriya Vidyalaya No.2 Kharagpur from
12/10 – 14/10/07
 Participated with National Cadet corps of India (Jadavpur University
Chapter): Participated in growing awareness and fighting against dengue
epidemic in slums and Kolkata and campus cleanliness drive to stay hygienic and
healthy 2017
 Kolkata Cycle Samaj (promoting non-motorized transport and a
pollution-free city) in association with Kankurgachi, Abhijan club:Certificate of participation for Kolkata round-trip bicycle rally held on January 15,
2017 from Kankurgachi to Tollygunj and back.
Sports:
 Cricket:
Participated and secured runners up position in under 16 cricket league at
Kendriya Vidyalaya Sangathan Regional sports meet 2011-12
 Shot put:
Secured first position in School annual day sports meet 2008
 Discuss throw:
Secured first position in School annual day sports meet 2008
Secured third position in School annual day sports meet 2009
Clubs & Societies:
 Member of Jadavpur University Photographic Club, one of the oldest
student-run club in Asia, 2016- 2018
 Peoples Film Collective, Kolkata
Festivals & Exhibitions:
 Participated in Google science fair, 2013
 Organised Jadavpur University Photo Festival, 2017 first of its kind in
Kolkata
 Kolkata Baul Fakir Utsav
 Kolkata International Film Festival
 Kolkata People’s Film Festival
 Tata Kolkata Literary Meet
 Apeejay Kolkata literary festival
 Kolkata literary festival, Kolkata International Book Fair
Interests:
Driving, cycling, travelling, reading articles, watching films, Playing football,
cricket, volleyball, chess, carom, badminton, table tennis"""

# questions based on stock context blob by hugging face

# question = "What does the sun provide to the earth?"
# question = "What is the sun's diameter?"

# new question based on the cv text i pasted in new context as blob
# question = "What is full name of Anirban?"
question = 'Experience in arts for Anirban ??'

response = model({ "question": question, "context": context })

print(question)
print("... Analyzing 🤖")
print("According to our AI, the answer to your question is:", response['answer']) # energy
