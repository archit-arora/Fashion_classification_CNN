##Scrapy script for searching Macys for images of different fashion categories  and downloading them to disk

import scrapy
import urllib
import time
import os
path=os.getcwd()+'/Datasets/images_macys_original'

class MacysSpider(scrapy.Spider):
    name = 'macys'
    def start_requests(self):
        self.MAX_COUNT = 200
        products_list=['Shirts','T-Shirts','Suits'] #define criteria for search on Macys website
        for product in products_list[:]:
            self.product=product.replace(' ','-')
            self.current_url='http://www1.macys.com/shop/featured/'+self.product #url obtained from product search
            self.count = 1
            self.page_index=30
            yield scrapy.Request(url=self.current_url, callback=self.search_parse)


    def search_parse(self, response):
        #extract urls of results obtained from image search to obtain high-res images of products 
        search_image_urls = response.css('.imageLink.productThumbnailLink.absolutecrossfade ::attr(href)').extract() 
        for image_page_url in search_image_urls[:]:
            image_page_url = response.urljoin(image_page_url)
            yield scrapy.Request(image_page_url,callback=self.image_parse) 
        if self.count>=self.MAX_COUNT:
            return
        self.page_index+=1
        next_url=self.current_url+'/Pageindex/'+str(self.page_index)
        yield scrapy.Request(url=next_url,callback=self.search_parse)

    def image_parse(self,response):
        image_page=response.css('.imageItem.selected ::attr(src)').extract()  #extract urls of high-res version of product images                      
        urllib.urlretrieve(image_page[0], path+'\\'+self.product+'\\'+str(self.count)+".jpg") #download images to disk using urllib
        self.count+=1
        return None

            






