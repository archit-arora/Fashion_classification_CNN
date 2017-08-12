import scrapy

class PostsSpider(scrapy.Spider):
    name = 'posts'

    def start_requests(self):
        self.count = 0
        self.max_count = 45
        urls = ['http://mat.tepper.cmu.edu/blog/']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    # Figure out how to do these queries using Selector Gadget, xpath
    # and scrapy shell 'http://mat.tepper.cmu.edu/blog/'
    def parse(self, response):
        dates = response.css('.published').re('title="(.*)">')
        titles = response.css('.entry-title a::text').extract()
        # Yielding a dictionary produces an "item"
        for date, title in zip(dates, titles):
            self.count +=1
            yield {'date': date, 'title ': title}
            if self.count >= self.max_count:
                return
        next_page = response.css('#nav-below a::attr(href)').extract_first()
        # In case it is a relative url
        next_page = response.urljoin( next_page )
        yield scrapy.Request(next_page, callback=self.parse)
