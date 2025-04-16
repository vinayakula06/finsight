'use client'

import { useEffect, useState } from 'react'
import Navigation from '@/components/Navigation'
import Chatbot from '@/components/Chatbot'
import {
  Card,
  Title,
  Text,
  TabList,
  Tab,
  TabGroup,
  TabPanels,
  TabPanel,
  AreaChart,
  DonutChart,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
} from '@tremor/react'
import { api } from '@/services/api'

interface PortfolioData {
  totalValue: number;
  dailyChange: number;
  dailyChangePercent: number;
  holdings: Array<{
    symbol: string;
    shares: number;
    avgPrice: number;
    currentPrice: number;
  }>;
  history: Array<{
    date: string;
    value: number;
  }>;
}

export default function Portfolio() {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await api.getPortfolioData();
        setPortfolioData(data);
      } catch (error) {
        console.error('Error fetching portfolio data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="lg:pl-72">
          <div className="px-4 py-10 sm:px-6 lg:px-8">
            <Text>Loading portfolio data...</Text>
          </div>
        </main>
      </div>
    );
  }

  const assetAllocation = [
    { name: 'Stocks', value: 45 },
    { name: 'Bonds', value: 30 },
    { name: 'Cash', value: 15 },
    { name: 'Crypto', value: 10 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="lg:pl-72">
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          <Title>Portfolio Dashboard</Title>
          <Text>Manage and track your investments</Text>
          
          <TabGroup className="mt-6">
            <TabList>
              <Tab>Overview</Tab>
              <Tab>Holdings</Tab>
              <Tab>Performance</Tab>
              <Tab>Analysis</Tab>
            </TabList>
            
            <TabPanels>
              <TabPanel>
                <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
                  <Card>
                    <Title>Portfolio Value Over Time</Title>
                    <AreaChart
                      className="mt-4 h-72"
                      data={portfolioData?.history || []}
                      index="date"
                      categories={["value"]}
                      colors={["blue"]}
                      valueFormatter={(number) => `$${number.toLocaleString()}`}
                    />
                  </Card>
                  
                  <Card>
                    <Title>Asset Allocation</Title>
                    <DonutChart
                      className="mt-4 h-72"
                      data={assetAllocation}
                      category="value"
                      index="name"
                      valueFormatter={(number) => `${number}%`}
                      colors={["blue", "cyan", "indigo", "violet"]}
                    />
                  </Card>
                  
                  <Card>
                    <Title>Quick Stats</Title>
                    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <Text>Total Value</Text>
                        <p className="text-2xl font-semibold text-blue-600">
                          ${portfolioData?.totalValue.toLocaleString()}
                        </p>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg">
                        <Text>Daily Change</Text>
                        <p className={`text-2xl font-semibold ${
                          portfolioData?.dailyChange >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {portfolioData?.dailyChange >= 0 ? '+' : ''}
                          ${portfolioData?.dailyChange.toLocaleString()} 
                          ({portfolioData?.dailyChangePercent}%)
                        </p>
                      </div>
                    </div>
                  </Card>
                </div>
              </TabPanel>
              
              <TabPanel>
                <Card className="mt-6">
                  <Title>Your Holdings</Title>
                  <Table className="mt-4">
                    <TableHead>
                      <TableRow>
                        <TableHeaderCell>Symbol</TableHeaderCell>
                        <TableHeaderCell>Shares</TableHeaderCell>
                        <TableHeaderCell>Avg Price</TableHeaderCell>
                        <TableHeaderCell>Current Price</TableHeaderCell>
                        <TableHeaderCell>Market Value</TableHeaderCell>
                        <TableHeaderCell>Gain/Loss</TableHeaderCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {portfolioData?.holdings.map((holding) => {
                        const marketValue = holding.shares * holding.currentPrice;
                        const costBasis = holding.shares * holding.avgPrice;
                        const gainLoss = marketValue - costBasis;
                        const gainLossPercent = ((gainLoss / costBasis) * 100).toFixed(2);

                        return (
                          <TableRow key={holding.symbol}>
                            <TableCell>{holding.symbol}</TableCell>
                            <TableCell>{holding.shares}</TableCell>
                            <TableCell>${holding.avgPrice.toFixed(2)}</TableCell>
                            <TableCell>${holding.currentPrice.toFixed(2)}</TableCell>
                            <TableCell>${marketValue.toLocaleString()}</TableCell>
                            <TableCell className={gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}>
                              {gainLoss >= 0 ? '+' : ''}${gainLoss.toLocaleString()} ({gainLossPercent}%)
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </Card>
              </TabPanel>
              
              <TabPanel>
                <Card className="mt-6">
                  <Title>Performance Metrics</Title>
                  <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <Text>1 Month Return</Text>
                      <p className="text-2xl font-semibold text-blue-600">+5.2%</p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <Text>3 Month Return</Text>
                      <p className="text-2xl font-semibold text-green-600">+12.8%</p>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <Text>YTD Return</Text>
                      <p className="text-2xl font-semibold text-purple-600">+18.5%</p>
                    </div>
                  </div>
                </Card>
              </TabPanel>
              
              <TabPanel>
                <Card className="mt-6">
                  <Title>Risk Analysis</Title>
                  <div className="mt-4 space-y-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <Text>Portfolio Beta</Text>
                      <p className="text-2xl font-semibold text-blue-600">1.2</p>
                      <p className="text-sm text-gray-600 mt-1">
                        Your portfolio is slightly more volatile than the market
                      </p>
                    </div>
                    <div className="p-4 bg-yellow-50 rounded-lg">
                      <Text>Diversification Score</Text>
                      <p className="text-2xl font-semibold text-yellow-600">7.5/10</p>
                      <p className="text-sm text-gray-600 mt-1">
                        Consider adding more diverse assets to reduce risk
                      </p>
                    </div>
                  </div>
                </Card>
              </TabPanel>
            </TabPanels>
          </TabGroup>
        </div>
      </main>

      <Chatbot />
    </div>
  );
} 