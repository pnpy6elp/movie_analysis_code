# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:48:59 2020

@author: 남준녕
"""
import numpy
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import json
import collections
import math
import re


def tree():
    return collections.defaultdict(tree)


# 약 0.54,0.46
# 8860, 7548

driver = webdriver.Chrome('./chromedriver_win32/chromedriver.exe')


# 117페이지 제일 마지막 두개 제외하기
# 381페이지 마지막 세개 제외하기
# 1028페이지 마지막 네개 제외하기
# 1029페이지부터 date 없앰
def rating_list_crawler(desired_):
    page_number = 1  # 암살
    jsonN = 0
    # trust = "T"
    while (page_number != desired_):

        for i in range(1, 11):
            try:
                driver.get(
                    "https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=121048&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}".format(
                        page_number))

                # f = open("./data5/review_history_gongjo.txt".format(((page_number - 1) * 10) + i), "a+t", encoding='utf-8')

                req = requests.get(driver.current_url)
                html = req.text
                soup = BeautifulSoup(html, 'html.parser')
                css_select = "body > div > div > div.score_result > ul > li:nth-child({}) > div.score_reple > dl > dt > em:nth-child(1) > a > span".format(
                    i)
                sympathy_select = soup.select(
                    "body > div > div > div.score_result > ul > li:nth-of-type({}) > div.btn_area > a._sympathyButton > strong".format(
                        i))[0]
                unsympathy_select = soup.select(
                    "body > div > div > div.score_result > ul > li:nth-of-type({}) > div.btn_area > a._notSympathyButton > strong".format(
                        i))[0]
                sym = int(sympathy_select.text)
                unsym = int(unsympathy_select.text)

                '''if(sym>unsym):
                    trust = "T"
                elif(sym<unsym):
                    trust = "UT"
                else:
                    trust = "None"'''
                user_name_list = soup.select(
                    'body > div > div > div.score_result > ul > li > div.score_reple > dl > dt > em > a > span')
                driver.find_element_by_css_selector(css_select).click()
                page = driver.current_url
                user_req = requests.get(page)
                user_html = user_req.text
                user_soup = BeautifulSoup(user_html, 'html.parser')

                h5 = user_soup.find('h5', attrs={'class': 'sub_tlt underline'})
                div = h5.find('div', attrs={'class': 'h5_right_txt'})
                number = div.find('strong')
                number0 = number.text
                number1 = number0.replace('<strong class="c_88 fs_11">', '')
                number2 = number1.replace('</strong>', '')
                number3 = float(number2)
                number4 = math.ceil(number3 / 10)
                page_num = 1
                user_name = h5.find(text=True, recursive=False)
                print(user_name)
                data_list = []
                rating = []
                while (page_num <= number4):
                    page = driver.current_url
                    user_req1 = requests.get(page + "&page=" + str(page_num))
                    user_html1 = user_req1.text
                    user_soup1 = BeautifulSoup(user_html1, 'html.parser')
                    table = user_soup1.find('table', attrs={'class': 'list_netizen'})
                    table_body = table.find('tbody')

                    rows = table_body.find_all('tr')
                    for row in rows:

                        title = row.find('td', attrs={'class': 'title'})
                        score = title.find('div')
                        text1 = title.find_all(text=True, recursive=False)
                        num_selector = row.find_all('td', attrs={'class': 'num'})
                        numlist = []
                        for i in num_selector:
                            numlist.append(i)
                        '''date = numlist[1].find_all(text=True,
                                                   recursive=False)
                        date = str(date)
                        date = date.rstrip(']')
                        date = date.rstrip('\'')
                        date = date.lstrip('[')
                        date = date.lstrip('\'')'''
                        rt = text1[3]
                        rt = re.sub("\t|\n", "", rt)
                        tt = title.find_all('a', attrs={'class': 'movie color_b'})

                        em = score.find_all('em')
                        i = 0

                        for a, b in zip(em, tt):
                            data = {}
                            data["title"] = b.text
                            data["rating"] = str(a.text)
                            data["review"] = rt
                            rating.append(int(a.text))
                            # data["date"] = str(date)

                            data_list.append(data)
                    if page_num < 1000:
                        page_num += 1
                    else:
                        page_num = number4 + 1
                # data_modi = ""
                data_modi = []
                dit = dict()
                for j in data_list:
                    data_modi.append(j)

                dit["data"] = data_modi
                dit["sympathy"] = sym
                dit["unsympathy"] = unsym
                rating_mean = numpy.mean(rating)
                rating_sd = numpy.std(rating)
                dit["rating mean"] = rating_mean
                dit["standard deviation"] = rating_sd

                num = len(dit["data"])
                if num >= 1:
                    '''f.write(str(sym))
                    f.write(",")
                    f.write(str(unsym))
                    f.write("\n")'''

                    jsonF = "C:/Users/User/Desktop/data_1/assassin_modi/%d.json" % jsonN
                    with open(jsonF, 'w', encoding='utf-8') as write_file:
                        json.dump(dit, write_file, ensure_ascii=False, indent=3)
                    jsonN += 1
            except Exception:
                pass

        page_number += 1


if __name__ == "__main__":
    rating_list_crawler(3964)