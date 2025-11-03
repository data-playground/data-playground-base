# %%

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# %%

# url = "https://www.linkedin.com/jobs/search-results/?keywords=data%20analytics&origin=SWITCH_SEARCH_VERTICAL"

# headers = {
#     'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
#     'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
#     'Cookie': 'bcookie="v=2&9363bb27-3613-47b3-88fc-eb41e753596f"; bscookie="v=1&202303242358160482442c-14ff-40ad-8d75-1e51d3f23f85AQGNOZkUlFsbHsNxwh0tbfyjZBJdFUsQ"; liap=true; JSESSIONID="ajax:1896546110745477250"; li_sugr=e37a32f2-0ada-49bd-8214-219c53b84ca1; li_theme=light; li_theme_set=app; _guid=ca4865f3-8663-459f-b723-a6c3c8c0af5d; _gcl_au=1.1.623995099.1749397231; dfpfpt=35d78705a3ab480987e299253926f0d7; li_at=AQEDARbC__wCHDkgAAABl1FXAm0AAAGY3dRmaE4AOs64RpCxby3g0qWG3D0K6SORbniNvY32CB9eGJaJgSB6H2WoQlw0MgqVsMfBke36AqxoKFpQSF_LXdojPaI4Cr-SYR2giQeuns8tT_BHsZFPNVwz; lang=v=2&lang=en-us; __cf_bm=iVyibkBFWRDQc.rtPmUFSydyvpb3HlzUvytl0HmFm9k-1755813871-1.0.1.1-BPbYcJu9NxF8MwTPd0whGWf68rgRE5uNKW0mAf6uolcmW8PUNA_rILeS9.aU907h1apTc4NdqBMZu8U9M9lxou2rk31UEKR.6D89qF4raFE; timezone=America/New_York; UserMatchHistory=AQJwH_KQrqDrUgAAAZjOqZ4Oq5AOI5XygGQfddJvA_-naraH5nRvD98SaGP2tW3YdXIvTzNU090z9gTMyTsCx_2_3ikkQiLvDDLytu1YpcRCQ9DQ2aUioIE9UeeXj829kRD0nFG00TePV_0R46zkWqLvyB8gIDdid-CkNPowaJPRDGNZN-hdFN9SM15g_L3JfvmJK5_GpbifvBf7Bvp1KWZ-_Qy-BuaROFOtC0JuU8WVgvtajCc6priPClVAKUwNXw5Hpvz-Y416H2QNgcBVSJakLX4BjFnMesS0REfCPSrDwcXcWfn4nEKaYe5aWk_zJR5MOneK4sWuuS2zUCumYQhMfTl985t7wnC-6Go6hd6V76ljaA; AnalyticsSyncHistory=AQI0gLZjcBZjfAAAAZjOqZ4OCcnzy0qJldh9jfq4slTfB_d1mA94W-KQeCE_eshtiQ14pExJ_dU7An44WAJK5w; lms_ads=AQGuRQdd4pGYwAAAAZjOqZ67Tw0Pn4xZsmmJc4FmVghFCMBKFjHJk0rEywn9I_0pDlfQzyaWLYh1WUi83SD25yEwyVoxHe2z; lms_analytics=AQGuRQdd4pGYwAAAAZjOqZ67Tw0Pn4xZsmmJc4FmVghFCMBKFjHJk0rEywn9I_0pDlfQzyaWLYh1WUi83SD25yEwyVoxHe2z; lidc="b=TB68:s=T:r=T:a=T:p=T:g=7393:u=1325:x=1:i=1755813880:t=1755865433:v=2:sig=AQHViCGBmwoB0eLusW_UhqmCzXpxgc1q"; fptctx2=taBcrIH61PuCVH7eNCyH0B9zcK90d%252bIeoo1r5v7Zc26CVTQB%252ftETUchwfkOWFHdFIx9CSGZJN0daNFfz%252b592dTHwiX4snJ0GwrMOAcTqEO9CWP1NrkjtZONJ7G1FQGTn4BrQkA7QzjimusuflG%252bnnVyNvxFE1MB77Uf4zeen1wv%252bisQYqBQNI0aCvchCOGPhJSyPMzBd7OGuk1dQrp4xep3%252fJ0WOabM7przqyN5tShgmAAQ3kQScm8o2NOAi0tRMlPS%252bnkmtyBuP8ZCiWZBv02hICtxkA1OPwIN6BnkQuwymvHSwMmzM3sXhQ92DB1cUXiG4zABWJiCCLRK%252f%252fg%252byNgWcxFQUYEigBBLbfJJcm%252fY%253d; AMCVS_14215E3D5995C57C0A495C55%40AdobeOrg=1; AMCV_14215E3D5995C57C0A495C55%40AdobeOrg=-637568504%7CMCIDTS%7C20322%7CMCMID%7C05514269074343007292307914273593431304%7CMCAAMLH-1756418687%7C7%7CMCAAMB-1756418687%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1755821087s%7CNONE%7CMCCIDH%7C-68741582%7CvVersion%7C5.1.1; aam_uuid=05337127257388008822362266584827160259'
# }

# # %%

# r = requests.get(url, headers=headers)
# r

# # %%

# soup = BeautifulSoup(r.content, 'html.parser')
# %%


# %%

from selenium import webdriver

# %%

url = "https://www.sonara.ai/dashboard"

driver = webdriver.Chrome()

# %%

driver.get(url)

driver.find_element('xpath', '//input[@name="email"]').send_keys("pedrohmanodemoura@gmail.com")
driver.find_element('xpath', '//input[@name="password"]').send_keys("j8gS4tL4@jz@Sw6")
driver.find_element('id', 'login-btn').click()

# %%

js_string = """
url = "https://www.sonara.ai/ea/api/v1/expert-apply/profiles/2954c26c-9895-b742-fc80-5592e09a47a2/jobs/suggested?page=1&size=100000&suggestedJobsType=Both&IncludeNonExpertJobs=false&datePosted=Any%20time"

headers = {
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-US,en;q=0.9,pt;q=0.8',
    'Cookies': document.cookie,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'
}

return fetch(url, {
    method: 'GET',
    headers: headers
})
    .then(response => response.json())
    .then(data => data);
"""
# %%

results = driver.execute_script(js_string)
# %%

### Built in NYC

# %%

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,pt;q=0.8",
    # "Cookie": "_ga=GA1.1.1848700245.1749417744; BIX_AUTH=chunks-2; BIX_AUTHC1=CfDJ8N49pBgB_XBPsL35S65RW-dGOlQuSIn-x8sCXNwyD-Y16XwrJSvVEXZhfBPOK-CJoX-bVOmXAlsH5sHO_JxKcvTogAT9hD8QhSGa6BWgPtG280k7rSyMijGF6m-7z7d1COvbibtEJgmRO9y_cLwFCJbHJtm6TiHLyIJ9H1nsNH4kNbXC-uYQF4iNT2khN5iM9F_cJJxXdv3R1_yHWWzv1dxI1BSYRGdBihphNYpOwiahKPZds96Aaym0YrtvBQG0H7pnJrnEg_w4u-vao6UdLfLV-7aCzRMUU5cS-xbPtr0B8D5aKh1yj73SENReQP7Q37WAgysA8Jg8CXy9yT7BcJtF5-VSEv4jfXlvRwK2aQq4h63n-1vka-HhIabM-W4zChD23iG_zgyGPWWVV9rvofOeOrH7-kCKxS7iHKQC-2IsdH_hIMgdDIov0xvJP47VibYcigRDQIJxpPKnt78UlhNh29PnpcuJ1u2qCmmpIjZr-ISd_KpdJ9Wg0aG312uhO7r2eGLiinC1Mgy4uWv-pdpQX3pUopkpCM0JSnfXBjGdISTAYTZjvYHVQ17vyfZpL4Lu7GTDAd0iMZGEN7e9O-BOmYBw2cZa128G3Koqs_52LdcMebKQHiqElrOGWYlNaFWUphOlJyqYFaj9aNs4d_fbkbSu-787Ri7Qc4KxN4WL0P_61U744eZtp0W-RhNYSOdaWGTVC9kIXAN-u0eMHNg38DOy6GZQ4o9TBFLjkazYFUBMyjUYK3G489MDvQwiSomqc-UOGOkQNVC0Ptmn1dYN4CgIblo5LM90Z0fXhml2qYb68QTyGlbxM5mH-gMiswkfi8hGBsUbp8gJQo3OyYR7JGs9EHbH9QZEwIVfIq7XiWSBR4aDqsxLW5K-DnQ6ubuOsmOJadeVA9SUXFsuCQ_RSYNyF0dIE1xe2h20f5VqjhVDs0f_hUssYaQARp2tY-JmPHTVyrbaP-4GZctOE53KF7qXuoBlKaRv7B9CJfVrh8KcGTcslXCXhiUiEIgMEE8lL193m4GL0R3D7VsKyBT4XWmMEKjOlPYwZIHKVPEFTZsoaBvpg7wQ7cStQD-z7x0SH4dar3FPoxhbMQr1Ap6fIojqsG3SPfmEO_ZQ_XU9bjPkZbERNWq4nU3CwwfAsdC0U2ZFYkfZQHMRux6sduSLtdZdDfRLwG1GaGml_9STzNmhWn4_7NToifQ88jKL_ALMkTKQLmxBKHg8hOP0X01cYVQf2TWbLIRt1z-8q1cMnO-8V1fMy1aTtzB0cd1UTyYmoeDiCGSX4T14g_SXGczmvJhQqNBaRJ7-R61hPpEIEf92uTtUeDE1WJAu5P4-wkCTQOGt-muBdCmlSezc6LaG0Pytxn5V4JYpuTowAoH2-pxoTAGyxfV-543sMfTpGUUWNenHesYkIQZ-O2PN9TRgPG_I7qets10NB4nhoCF--B4GB3lvzN8XgidJoJUt76lbrgHCLPWhyOa5YkI5uUTicnzIlCUjS5UiojiET6WzhCOsG1hQjmlKCvdKAOCCLwdkcBs_6WP86RG2p8FD0nLBC3bMHArWx28jnk1c3FxoxgLgIc9jYc-Ld1pTxKbXGlcg3_L5clFE9Zm7d0SCVJ3gONCYAP0fR6M6FIz1wAtY-muQftW5b3h8V-eYD2AW1R6cvinqDmN5Yb-htF_OEk8RKB2LKe7M5dJd2zZhwHv6EFcQ8aMaxOaxATPCL4InDtc8uBD_yF3Ctq9UqRWq6CbtFc3ch-1ate4UTKS-dXJInCtzTQCBkBmcWyr2CwXV9siYIbTbvwqA79M76MQuyJ8kZWr6OsSkKaxOKw4t1ZCMYOAfK1WUe-pWq2Y3r41kV8I9-OJoDh-8VVdFkQxUWlvF1hQT9cX1XR4x7zI2ToZsC5L5hgb2XhK9z3E23gr9q0s7KJTftyIVZjog4oufan7Sle1Tj7NG3wXVVZcRD50TzqKgz2gawtQN7nQ2lVkw3Ht1CIP6cxjDRWz2f7qWpH3HVwK_WmfMrVtA-nlg4_uL5AAP7DhrFu5slnGcVGQXV-ODA-ZZbYXEmVp0RpULCwp7XgyzMyV1jJ2tWsdV3H5YXDRj3vsOxoak3OxID-RxikkrpMy5ctZzvfzEKC-wMGdzO3a1pzQOBoiKrfZpeUBpSanhUEbOmY97GRpG-mp1ynXQnvw1nAFYoN_LLoLHPWyOTk4O5IniliM_lTkRLXJWvLUa0Oz2lpXfibULTv05Mdck9_6mp4unt9N9Ujpx46jnaPQoErdLzG1ep_NBXbrYk_DsmiNZtYfiK6hhA3iAy2W8bzn0-MUeRNq7c-ala923WgtIYaNHxuoBegB7J5aAwpasW_wWt333Z5HGCLJfKKXBKAFIbukWTxaBrk6WEvM0NY4gaygRDuR8leVlw8akOvPbQiCTyHM6ShhMKZK5qRfPxTABwr23qYH3fY19hMAbEHRAr2dzXIj33AtvIKCgGWgckjbb0Y-NXXbz3eBDjGrzW8sP1-fIvaAIUNMj91ObwxlG2ZG5lfTusUWVaw_65nUHrxGUVv6kkKKM0Wh2DkgJBlQtOPsn069gtimk5dIja0Lt-N7GVVzIo5NBI_t5tDJKPQjT6YbChkEusJ0F-52Eysrk9AsUtqLfmTma3xVEOkAarv0wX2LC8BdLqj1DYEoAWfkwjh2j-x9JXjsFKPGMZwX-UH33s92n_et3hH4FEynR_Q8u50LMH5Sz3iEY5xjtP_HBJnm6BgsQJTGRaE1PBOwUMef-LHktYHZNQ9BQcljOWcuu3qFQf6TjY8reXsPt0CZk5DLr5bHnSsvVa9GFwKznOj74gWBcWaF8PsCf0cXcB5DG-VVqOA7Yl6rkfWRZWWe3p6vikHchsvO8AKZby9etiR4WklEAGiR-g1XY0A_dRLhxKtw5f9msrmgEWd0aQsS1jSF0A3luQKhZE06UeBvZzctBaf3HIJvqIoM2GDzGI0A1fTetc9l70v8v_Zpq5eLoGBicXXW7iwXVkxI3W9Xq1uDi5BolWIPxSVr-zhKAm19YOOTjOQEB4maM65jbOgktNgi5FxlDN6ANN4b-hxFNCqrtWiq4ecTbQ6lQ6cdqTMqRSn4P1S_Dn6pipSInKq0_H7EJqB_7xG66BO_pJuTWhNL1fN3JK3F6_Xhgpw81U0H7WrQD9iEpaT1J4oquBKIUB4012iw3PRwQFg-Gf7m4IlrUdd1JnrqDqgvJ6opFPnSsZeVZF0plYxagUpCh-JdxbTrBaD7IHdkFw1V1lyB2ZFxOcDzrLs4yKPAYPxxI0krfF5yD0JPpV9oD6Ioy1C-4d_ULWi4JDPXl-kHwYba-P4p43wf8g7k7n5-sz9nDF175EQ7p4P6xMLGU66VPkenR_hwOy8-rFwXqJ9p4GcQeZwCMcZD25rfU02s51zfbMYTEBHJlDRmDGUePMYws6nZchxAdtFC27iYobyoaUlB9_Jx6NXjoqwvr_7e-ZgKEHlKAL03zql2nrt4n2Yf36vw0ipd4ykwf8UYXCyFTYLYxjmCuOq3mm4e9ppqUhLp5SXQMeni1gW5OqKQR_Y7UdgpXHpj5pxwpLdQ_qWFnnSOFTdvGSVE2-ZHKb4mJo0vXxcvcYaW34MKoaKEzGXWPhUKdga7R-guKThZ1_860d4B4v_X3DF_HhxB-oGXriDpp2OfHd61napMFKa6EYWw-4hXESL5UijXskYbFvN9JvfHexase3uldaT5dLikMD6DAqbGkYJzBhXH8wJer_5D9T84T5eyhRzs4O7CDms5_VX3nYv-CzZ5y_XxFWxlI9NE0RypItj4QRzgRij7s8ngwGzYSbzdqe_ns6HTc3V-n7VJg1rTgZO9zz33q8CcPyF5gY0OZfPVJUJCYXnrEPWdb0V_BwLxIPKB9LeeueleEk06hk_IR_AxDZ-C9KCN7rjOORas_42nYau6tLNoIJrVgu3M; BIX_AUTHC2=coYrA6ekdQKXXmd1RoryiHo9ERu5Zdxd1833aexZWgHWPysFQpP06U63WdZF7Yl3s36W2y8Sc7NUPr3VXgaDdD9ec16D6yHgBfb3RAeTdFOcZ0hVo30x-lEY6ZefUeXbzeF5XI4_7nYFscarJ6Pmorl3iNL0TNym2KtjAO1KmK3cLN7nYTh7lnQL6GuH9vtXBL7lI0vFBpbCXasMyaITP7eokX0vct8uo7AwCaxZqYPronBHH0RDAgONh6afjjSlaujDCp8pR3r1jFzYot596krAgzqidcuFQbnniPeg4kxzndPApZOMGxp_5uUq2KCDXB_qUboAwffv8QkF1AVU5rOBLUqKxOvkWp9NrkITe5CpEbet_EYV_Ta4QOt8l0nP2UlwJRDUuCOwKIATECdBPblieHpQAinxp7lppZFld7S35_knW9JB_rchExBfumU3ZHnfVKxfmRZlftyGA58Bs0QRMjuFHGQpnndgOSuNmpPN6rnYgEMTOcUcTqXvLIL76gV5egt6GWqVQ6EKKTQPIwXJmUrM5KKrL4vDk5p68rkVidyuZZjg1td-fv185gw2UziJAE7kKxgQJMvQkHDbZY1AFQwBmTzDdEV60zwpgV9fLb9EGypJGY4vp76DRbaMCF4_qn_w186ernQE3OVGLItynxIy3SSeUXgJTjo-yCIQ5qnH0nyPFy-kBwi6EdN-3Nk1NZ-ejkC6-UA; clientId=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjIwMTc4MGU4LTQ3NTctNDFiZC1iOTNiLWQ0YjQwZDY1Yzg0YSJ9.P4u59dbPhSnvpjwLCdXCK8IhvamjgDlrzP48Ojfhfmc; _ga_PD9P2L7TK7=GS2.1.s1756656227$o2$g1$t1756656411$j59$l0$h0; OptanonConsent=isGpcEnabled=0&datestamp=Sun+Aug+31+2025+12%3A06%3A51+GMT-0400+(Eastern+Daylight+Time)&version=202503.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0004%3A1%2CC0002%3A1&AwaitingReconsent=false",
    "Referer": "https://www.builtinnyc.com/",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
}
# %%

def get_jobs(page):
    url = f"https://www.builtinnyc.com/jobs/data-analytics/senior/expert-leader?skills=Python,Excel,SQL,AI,GCP,JavaScript,Tableau,MS%20Office&city=New%20York%20City&state=New%20York&country=USA&allLocations=true&page={page}"

    r = requests.get(url, headers=headers)

    soup = BeautifulSoup(r.content, 'html.parser')

    jobs = soup.find_all("div", attrs={"data-id": "job-card"})

    job_details = [get_details(job) for job in jobs]

    num_pages = soup.select('.pagination > li')[-2].text

    return job_details, int(num_pages)

# %%

def get_details(job):
    job_details = {
        "company_title": "",
        "job_title": "",
        "job_link": "",
        "reposted": None,
        "remote_option": None,
        "location": None,
        "easy_apply": False,
        "salary": None,
        "level": None,
        "categories": None,
        "summary": None,
        "skills": None
    }

    job_details["company_title"] = job.find("a", attrs={"data-id": "company-title"}).text.strip()
    job_details["job_title"] = job.find("a", attrs={"data-id": "job-card-title"}).text.strip()
    job_details["job_link"] = "https://www.builtinnyc.com" + job.find("a", attrs={"data-id": "job-card-title"}).get("href").strip()

    attributes = job.select(".bounded-attribute-section > div > div > div")

    for attr in attributes:
        if attr.find(class_="fa-clock"):
            job_details["reposted"] = attr.find(class_="fa-clock").parent.text.strip()
        elif attr.find(class_="fa-house-building"):
            job_details["remote_option"] = attr.text.strip()
        elif attr.find(class_="fa-location-dot"):
            if attr.find("span", attrs = {'data-bs-toggle': "tooltip"}):
                job_details["location"] = " • ".join([i.text for i in BeautifulSoup(attr.find("span", attrs = {'data-bs-toggle': "tooltip"}).get('data-bs-title'), 'html.parser').find_all('div')])
            else:
                job_details["location"] = attr.text.strip()
        elif attr.find(class_="fa-bolt"):
            job_details["easy_apply"] = True
        elif attr.find(class_="fa-sack-dollar"):
            job_details["salary"] = attr.text.strip()
        elif attr.find(class_="fa-trophy"):
            job_details["level"] = attr.text.strip()

    job_details['categories'] = job.find(class_ = "mb-md fs-xs fw-bold").text.strip()
    job_details['summary'] = job.find(class_ = "fs-sm fw-regular mb-md text-gray-04").text.strip()
    job_details['skills'] = " • ".join([i.text for i in job.select(".d-md-inline.ps-md-sm > .fs-xs.text-gray-04.mx-sm")])

    return job_details

def get_job_description(s, url):
    r = s.get(url, headers=headers)

    soup = BeautifulSoup(r.content, 'html.parser')

    description_body = soup.find("div", class_ = "html-parsed-content")

    return "\n\n".join([i.get_text() for i in description_body.children])

# %%

job_details = []
num_pages = 1000
page = 1

while page <= num_pages:
    print(f"Page: {page}")
    
    page_results, num_pages = get_jobs(page)

    job_details.extend(page_results)

    page += 1

# %%

# for job in tqdm(job_details):
#     s = requests.Session()

#     try:
#         job['description'] = get_job_description(s, job['job_link'])
#     except Exception as e:
#         print(f"Error: {e}")
#         job_details['description'] = None
# %%
