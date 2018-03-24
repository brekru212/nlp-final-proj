import datetime
import csv

def getprices():
    prices = {}
    with open('price_history/AAPL.csv', 'rb') as csvfile:
        price_history = csv.DictReader(csvfile)
        for row in price_history:
            prices[row['Date']] = row
    return prices
    # for k,v in prices.iteritems():
    #     print k
    #     print v

prices = getprices()

class Doc8K():
    def __init__(self, date, company, text, events):
        self.date = datetime.date(date[0], date[1], date[2])
        print str(self.date)
        self.company = company
        self.text = text
        self.events = events
        self.bigrams = {}
        self.unigrams = {}
        self.preprices = []
        self.postprices = []
        self.doc_price = prices[str(self.date)]
        # print self.doc_price
        self.getpre()
        self.getpost()
        # print self.preprices

    def getpre(self):
        n = 1
        days_bak = n
        while n < 6:
            try:
                cur_date = self.date + datetime.timedelta(days= -1 * days_bak)
                # print str(cur_date)
                # print prices[str(cur_date)]
                self.preprices.append(prices[str(cur_date)])
            except KeyError:
                days_bak += 1
                continue
            n += 1

    def getpost(self):
        n = 1
        days_bak = n
        while n < 6:
            try:
                cur_date = self.date + datetime.timedelta(days=days_bak)
                # print str(cur_date)
                # print prices[str(cur_date)]
                self.postprices.append(prices[str(cur_date)])
            except KeyError:
                days_bak += 1
                continue
            n += 1




def open_8k():
    doc_list = []
    f = open('8K-gz/AAPL', 'r')
    for l in f:
        if l == '\n':
            continue
        # print type(l)
        # print len(l)
        if '<DOCUMENT>' in l or 'ITEM:' in l:
            cur_doc = []
        elif '</DOCUMENT>' in l:
            # print l
            try:
                new_doc = Doc8K(doc_date, 'APPL', cur_doc, eve_list)

            # print new_doc
            # print cur_doc
                doc_list.append(new_doc)
                cur_doc = []
            except KeyError:
                print 'hello world'
                cur_doc = []
        elif 'TIME:' in l:
            # print l
            time = l[5:]
            year = time[0:4]
            month = time[4:6]
            day = time[6:8]
            # doc_date = datetime.date(int(year), int(month), int(day))
            # doc_date = year+"-"+day+"-"+month
            doc_date = (int(year), int(month), int(day))
            # print year
            # print month
            # print day
        elif 'EVENTS:' in l:
            events = l[7:]
            eve_list = events.split('\t')
            eve_list = [value.strip() for value in eve_list if value != '']
            # print eve_list
        else:
            cur_doc.append(l)
    print len(doc_list)

    #     print l
    #     cur_doc = []
    #     if '<DOCUMENT>' in l:
    #         doc_list.append(cur_doc)
    #         cur_doc = [l]
    #     else:
    #         cur_doc.append(l)
    # print doc_list

open_8k()

