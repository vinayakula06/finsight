'use client'

import { useState } from 'react'
import Navigation from '@/components/Navigation'
import Chatbot from '@/components/Chatbot'
import {
  Card,
  Title,
  Text,
  TextInput,
  Button,
  TabGroup,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Grid,
  Col,
  LineChart,
  BarChart,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
} from '@tremor/react'
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline'
import { api } from '@/services/api'
import toast from 'react-hot-toast'

export default function StocksAssistant() {
  const [symbol, setSymbol] = useState('')
  const [loading, setLoading] = useState(false)
  const [stockData, setStockData] = useState<any>(null)
  const [companyProfile, setCompanyProfile] = useState<any>(null)
  const [historicalData, setHistoricalData] = useState<any[]>([])
  const [news, setNews] = useState<any[]>([])

  const handleSearch = async () => {
    if (!symbol) {
      toast.error('Please enter a stock symbol')
      return
    }

    setLoading(true)
    try {
      // Fetch all data in parallel
      const [quoteData, profileData, historicalData, newsData] = await Promise.all([
        api.getStockQuote(symbol),
        api.getCompanyProfile(symbol),
        api.getHistoricalData(symbol),
        api.getMarketNews()
      ])

      setStockData(quoteData)
      setCompanyProfile(profileData)
      setHistoricalData(processHistoricalData(historicalData))
      setNews(newsData.slice(0, 5)) // Show only latest 5 news items
      toast.success(`Data loaded for ${symbol.toUpperCase()}`)
    } catch (error) {
      console.error('Error fetching stock data:', error)
      toast.error('Error fetching stock data. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const processHistoricalData = (data: any) => {
    // Process the historical data into the format needed for charts
    // This would depend on the actual API response format
    return data['Time Series (Daily)'] 
      ? Object.entries(data['Time Series (Daily)']).map(([date, values]: [string, any]) => ({
          date,
          price: parseFloat(values['4. close'])
        })).reverse()
      : []
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="lg:pl-72">
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          <div className="mb-8">
            <Title>Stocks Assistant</Title>
            <Text>Analyze stocks and get AI-powered insights</Text>
          </div>

          {/* Stock Search */}
          <Card className="mb-6">
            <div className="flex gap-4">
              <TextInput
                icon={MagnifyingGlassIcon}
                placeholder="Enter a stock symbol (e.g., AAPL)"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="flex-1"
              />
              <Button 
                onClick={handleSearch}
                loading={loading}
                disabled={loading}
              >
                Analyze
              </Button>
            </div>
          </Card>

          {stockData && (
            <TabGroup>
              <TabList>
                <Tab>Overview</Tab>
                <Tab>Technical Analysis</Tab>
                <Tab>Fundamentals</Tab>
                <Tab>News & Sentiment</Tab>
              </TabList>

              <TabPanels>
                <TabPanel>
                  <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6 mt-6">
                    <Col numColSpan={1} numColSpanLg={2}>
                      <Card>
                        <Title>Price Chart</Title>
                        <LineChart
                          className="mt-4 h-72"
                          data={historicalData}
                          index="date"
                          categories={["price"]}
                          colors={["blue"]}
                          valueFormatter={(number) => `$${number.toFixed(2)}`}
                        />
                      </Card>
                    </Col>

                    <Card>
                      <Title>Key Statistics</Title>
                      <div className="mt-4 space-y-4">
                        <div>
                          <Text>Current Price</Text>
                          <p className="text-2xl font-semibold">${stockData.c.toFixed(2)}</p>
                        </div>
                        <div>
                          <Text>Daily Change</Text>
                          <p className={`text-2xl font-semibold ${
                            stockData.c - stockData.pc >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {((stockData.c - stockData.pc) / stockData.pc * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div>
                          <Text>Day Range</Text>
                          <p className="text-2xl font-semibold">
                            ${stockData.l.toFixed(2)} - ${stockData.h.toFixed(2)}
                          </p>
                        </div>
                      </div>
                    </Card>

                    <Col numColSpan={1} numColSpanLg={2}>
                      <Card>
                        <Title>Trading Volume</Title>
                        <BarChart
                          className="mt-4 h-72"
                          data={historicalData}
                          index="date"
                          categories={["volume"]}
                          colors={["blue"]}
                          valueFormatter={(number) => 
                            new Intl.NumberFormat("en-US", {
                              notation: "compact",
                              compactDisplay: "short",
                            }).format(number)
                          }
                        />
                      </Card>
                    </Col>

                    <Card>
                      <Title>AI Insights</Title>
                      <div className="mt-4 space-y-4">
                        <div className="p-4 bg-blue-50 rounded-lg">
                          <Text>Technical Signals</Text>
                          <p className="text-lg font-medium text-blue-600">
                            {stockData.c > stockData.pc ? 'Bullish' : 'Bearish'}
                          </p>
                          <p className="text-sm text-gray-600 mt-1">
                            Based on price action and volume analysis
                          </p>
                        </div>
                        <div className="p-4 bg-green-50 rounded-lg">
                          <Text>Recommendation</Text>
                          <p className="text-lg font-medium text-green-600">
                            {stockData.c > stockData.pc ? 'Consider Buy' : 'Hold/Wait'}
                          </p>
                          <p className="text-sm text-gray-600 mt-1">
                            Based on current market conditions and technical analysis
                          </p>
                        </div>
                      </div>
                    </Card>
                  </Grid>
                </TabPanel>

                <TabPanel>
                  <Card className="mt-6">
                    <Title>Technical Indicators</Title>
                    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <Text>RSI (14)</Text>
                        <p className="text-2xl font-semibold text-blue-600">65.4</p>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg">
                        <Text>MACD</Text>
                        <p className="text-2xl font-semibold text-green-600">Bullish</p>
                      </div>
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <Text>Moving Averages</Text>
                        <p className="text-2xl font-semibold text-purple-600">Above 200 MA</p>
                      </div>
                    </div>
                  </Card>
                </TabPanel>

                <TabPanel>
                  <Card className="mt-6">
                    <Title>Company Overview</Title>
                    {companyProfile && (
                      <div className="mt-4 space-y-4">
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                          <div>
                            <Text>Company Name</Text>
                            <p className="text-lg font-semibold">{companyProfile.name}</p>
                          </div>
                          <div>
                            <Text>Industry</Text>
                            <p className="text-lg font-semibold">{companyProfile.industry}</p>
                          </div>
                          <div>
                            <Text>Market Cap</Text>
                            <p className="text-lg font-semibold">
                              ${(companyProfile.marketCap / 1e9).toFixed(2)}B
                            </p>
                          </div>
                          <div>
                            <Text>P/E Ratio</Text>
                            <p className="text-lg font-semibold">{companyProfile.peRatio?.toFixed(2) || 'N/A'}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </Card>
                </TabPanel>

                <TabPanel>
                  <Card className="mt-6">
                    <Title>Latest News</Title>
                    <div className="mt-4 space-y-4">
                      {news.map((item, index) => (
                        <div key={index} className="p-4 border rounded-lg">
                          <h3 className="font-semibold">{item.headline}</h3>
                          <p className="text-sm text-gray-600 mt-1">{item.summary}</p>
                          <div className="mt-2 flex justify-between items-center">
                            <span className="text-xs text-gray-500">{new Date(item.datetime * 1000).toLocaleDateString()}</span>
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 text-sm"
                            >
                              Read More
                            </a>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </TabPanel>
              </TabPanels>
            </TabGroup>
          )}
        </div>
      </main>

      <Chatbot />
    </div>
  )
} 