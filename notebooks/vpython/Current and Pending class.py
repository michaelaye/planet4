
# coding: utf-8

# In[ ]:

POCs = {'SSW':'Mary Voytek [202-358-1577; mvoytek@hq.nasa.gov]',
        'MDAP':'Mitchell Schulte [202-358-2127; HQ-MDAP@mail.nasa.gov]'}

class Support(object):
    columns = ['Project Title', 'PI',
               'Sponsor & Program','Point of Contact',
               'Period of Performance',
               'Person-Months [WM]','FTE']
    status = {0:'Pending', 1:'Current'}
    
    def __init__(self, title, pi, sponsor, pop, fte, status=0):
        self.title = title
        self.pi = pi if pi!='me' else "K.-Michael Aye"
        self.sponsor = 'NASA ' + sponsor
        if pi!='me':
            self.sponsor += ' (as Co-I)'
        self.poc = POCs[sponsor]
        self.pop = pop
        self.fte = fte
        
    @property
    def PI(self):
        return self.columns['pi']
        
    @property
    def wm(self):
        return round(self.fte*12)
    
    @property
    def series(self):
        return pd.Series([self.title, self.pi, self.sponsor,
                             self.poc, self.pop, self.wm, self.fte],
                            index=self.columns)


# In[ ]:

s1 = Support('Cryo-venting', 'me', 'SSW', 'date1 to date2', 0.25).series


# In[ ]:

s2 = Support('DLA', 'Anya', 'MDAP', 'date1 to date2', 0.13).series


# In[ ]:

df = pd.DataFrame([s1, s2])


# In[ ]:

df.T


# In[ ]:

df.T.to_latex('test_latex_out.tex', index=False)


# In[ ]:

0.25+0.25+0.25+0.08+0.25


# In[ ]:



