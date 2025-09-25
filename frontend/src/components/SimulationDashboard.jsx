import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import {
  Upload,
  Download,
  TrendingUp,
  Activity,
  Users,
  Settings,
  ChevronDown,
  ChevronUp,
  FileSpreadsheet,
  Brain,
  Zap,
  Target,
  RefreshCw,
  Grid,
  List,
  AlertTriangle,
  Database,
  Globe,
} from 'lucide-react';
import FirecrawlPanel from './FirecrawlPanel';
import ModelToggle from './ModelToggle';

const palette = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#f97316', '#ef4444'];

const metricIcons = {
  'Total Projections': TrendingUp,
  'Sim Iterations': Activity,
  'Optimized Lineups': Target,
  'Avg. Win %': Users,
};

const SimulationDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [sites, setSites] = useState([]);
  const [selectedSite, setSelectedSite] = useState('');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [viewMode, setViewMode] = useState('grid');
  const [expandedCard, setExpandedCard] = useState(null);
  const [sliderValues, setSliderValues] = useState({
    correlation: 0.65,
    uniqueness: 0.5,
    upside: 0.8,
    ownership: 0.55,
    variance: 0.4,
  });
  const [workflowProgress, setWorkflowProgress] = useState([]);
  const [workflowResults, setWorkflowResults] = useState(null);
  const [showUploadInstructions, setShowUploadInstructions] = useState(false);
  const [modelConfig, setModelConfig] = useState({
    use_pytorch: false,
    pytorch_blend_weight: 0.5,
    model_type: 'ridge_baseline'
  });

  const fetchSites = useCallback(async () => {
    try {
      const response = await fetch('/api/dashboard/sites');
      if (!response.ok) {
        throw new Error('Unable to load supported sites');
      }
      const payload = await response.json();
      const siteList = payload.sites || [];
      setSites(siteList);
      if (siteList.length && !selectedSite) {
        setSelectedSite(siteList[0]);
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    }
  }, [selectedSite]);

  const fetchDashboard = useCallback(
    async (site) => {
      if (!site) {
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`/api/dashboard/site/${site}`);
        if (!response.ok) {
          throw new Error('Dashboard data is not available yet. Run the CLI tools to generate CSV output.');
        }
        const payload = await response.json();
        setDashboardData(payload);
      } catch (err) {
        console.error(err);
        setDashboardData(null);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    fetchSites();
  }, [fetchSites]);

  useEffect(() => {
    if (selectedSite) {
      fetchDashboard(selectedSite);
    }
  }, [selectedSite, fetchDashboard]);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    setUploadedFile(file);
    setIsProcessing(true);
    // Upload handling is intentionally client side only; the data pipeline already populates the CSV inputs
    setTimeout(() => {
      setIsProcessing(false);
    }, 1200);
  };

  const handleSliderChange = (key, value) => {
    setSliderValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleRefresh = async () => {
    if (!selectedSite) {
      return;
    }
    setIsRefreshing(true);
    await fetchDashboard(selectedSite);
    setIsRefreshing(false);
  };

  const handleRunOptimizer = async () => {
    if (!selectedSite) {
      setError('Please select a site first');
      return;
    }

    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await fetch('/api/run/optimizer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          site: selectedSite,
          num_lineups: 20,
          num_uniques: 1,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to run optimizer');
      }

      const result = await response.json();
      console.log('Optimizer result:', result);
      
      // Refresh dashboard data after optimizer completes
      await fetchDashboard(selectedSite);
      
    } catch (err) {
      console.error('Error running optimizer:', err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunSimulation = async () => {
    if (!selectedSite) {
      setError('Please select a site first');
      return;
    }

    setIsProcessing(true);
    setError(null);
    
    try {
      // Choose API endpoint based on model configuration
      const endpoint = modelConfig.use_pytorch ? '/api/run/pytorch-simulation' : '/api/run/simulation';
      
      const requestBody = {
        site: selectedSite,
        field_size: 100,
        num_iterations: 1000,
        use_contest_data: false,
        use_file_upload: false,
        ...modelConfig // Include model configuration
      };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Failed to run simulation');
      }

      const result = await response.json();
      console.log('Simulation result:', result);
      
      // Refresh dashboard data after simulation completes
      await fetchDashboard(selectedSite);
      
    } catch (err) {
      console.error('Error running simulation:', err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunCompleteWorkflow = async () => {
    if (!selectedSite) {
      setError('Please select a site first');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setWorkflowProgress([]);
    setWorkflowResults(null);
    setShowUploadInstructions(false);

    // Get config values from inputs
    const numLineups = document.getElementById('quick-play-lineups')?.value || 20;
    const numIterations = document.getElementById('quick-play-iterations')?.value || 1000;
    const contestUrl = document.getElementById('quick-play-url')?.value || '';

    try {
      setWorkflowProgress(['🚀 Starting complete workflow...']);

      const response = await fetch('/api/run/complete-workflow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          site: selectedSite,
          num_lineups: parseInt(numLineups),
          num_iterations: parseInt(numIterations),
          contest_url: contestUrl,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to run complete workflow');
      }

      const result = await response.json();
      setWorkflowResults(result);
      setWorkflowProgress(prev => [...prev, '✓ Workflow completed successfully!']);
      
      // Refresh dashboard data
      await fetchDashboard(selectedSite);
      
    } catch (err) {
      console.error('Error running complete workflow:', err);
      setError(err.message);
      setWorkflowProgress(prev => [...prev, `✗ Error: ${err.message}`]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadResults = async () => {
    try {
      const response = await fetch('/api/download/lineups', {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error('Failed to download results');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'dk-lineups.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Error downloading results:', err);
      setError(err.message);
    }
  };

  const handleShowUploadInstructions = () => {
    setShowUploadInstructions(!showUploadInstructions);
  };

  const handleModelChange = (newModelConfig) => {
    setModelConfig(newModelConfig);
  };

  const compositionData = useMemo(() => {
    if (!dashboardData?.compositionChart?.length) {
      return [];
    }
    return dashboardData.compositionChart.map((item, index) => ({
      ...item,
      color: palette[index % palette.length],
    }));
  }, [dashboardData]);

  const projectionSample = useMemo(() => {
    if (!dashboardData?.projectionSample?.length) {
      return [];
    }
    return dashboardData.projectionSample.map((item) => ({
      ...item,
      own: item['own%'],
    }));
  }, [dashboardData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white">
      <header className="bg-black/40 backdrop-blur-xl border-b border-white/10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-8">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                  SimProjector Pro
                </h1>
              </div>

              <nav className="flex space-x-1">
                {['quick-play', 'dashboard', 'simulations', 'projections', 'optimizer', 'analytics', 'web-scraper'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-2 rounded-lg transition-all duration-200 capitalize ${
                      activeTab === tab
                        ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    {tab === 'web-scraper' ? (
                      <div className="flex items-center space-x-2">
                        <Globe className="w-4 h-4" />
                        <span>Web Scraper</span>
                      </div>
                    ) : tab === 'quick-play' ? (
                      <div className="flex items-center space-x-2">
                        <Zap className="w-4 h-4" />
                        <span>Quick Play</span>
                      </div>
                    ) : (
                      tab
                    )}
                  </button>
                ))}
              </nav>
            </div>

            <div className="flex items-center space-x-4">
              <ModelToggle onModelChange={handleModelChange} />

              <select
                value={selectedSite}
                onChange={(event) => setSelectedSite(event.target.value)}
                className="bg-white/5 border border-white/10 text-white px-4 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
              >
                {sites.length === 0 && <option value="">Select site</option>}
                {sites.map((site) => (
                  <option key={site} value={site}>
                    {site.toUpperCase()}
                  </option>
                ))}
              </select>

              <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors" onClick={handleRefresh}>
                {isRefreshing ? <RefreshCw className="w-5 h-5 text-indigo-400 animate-spin" /> : <Settings className="w-5 h-5 text-gray-400" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="bg-black/30 backdrop-blur-lg border-b border-white/5 px-6 py-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center space-x-4">
            <label className="relative cursor-pointer">
              <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload} className="hidden" />
              <div className="flex items-center space-x-2 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 rounded-lg transition-colors">
                <Upload className="w-4 h-4 text-white" />
                <span className="text-white font-medium">Attach CSV</span>
              </div>
            </label>

            {uploadedFile && (
              <div className="flex items-center space-x-2 px-3 py-1.5 bg-green-500/10 border border-green-500/30 rounded-lg">
                <FileSpreadsheet className="w-4 h-4 text-green-400" />
                <span className="text-green-400 text-sm">{uploadedFile.name}</span>
                {isProcessing ? <RefreshCw className="w-4 h-4 text-green-400 animate-spin" /> : <Database className="w-4 h-4 text-green-400" />}
              </div>
            )}

            <button
              onClick={handleRefresh}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-purple-500 hover:bg-purple-600 transition-all"
            >
              <RefreshCw className={`w-4 h-4 text-white ${isRefreshing ? 'animate-spin' : ''}`} />
              <span className="text-white font-medium">Sync Latest Output</span>
            </button>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center bg-white/5 rounded-lg p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded ${viewMode === 'grid' ? 'bg-white/10 text-white' : 'text-gray-400'}`}
              >
                <Grid className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded ${viewMode === 'list' ? 'bg-white/10 text-white' : 'text-gray-400'}`}
              >
                <List className="w-4 h-4" />
              </button>
            </div>

            <button className="flex items-center space-x-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg transition-colors" onClick={handleRefresh}>
              <Download className="w-4 h-4 text-gray-400" />
              <span className="text-gray-400 font-medium">Refresh</span>
            </button>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Quick Play Tab */}
        {activeTab === 'quick-play' && (
          <div className="space-y-6">
            <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center">
                <Zap className="w-6 h-6 mr-3 text-yellow-400" />
                One-Click DFS Lineup Generator
              </h2>
              <p className="text-gray-300 mb-6">
                Automatically fetch slate data, generate projections, and create optimized lineups ready for upload to DraftKings.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">Configuration</h3>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Number of Lineups</label>
                    <input
                      type="number"
                      min="1"
                      max="150"
                      defaultValue="20"
                      className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                      id="quick-play-lineups"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Simulation Iterations</label>
                    <input
                      type="number"
                      min="100"
                      max="5000"
                      defaultValue="1000"
                      className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                      id="quick-play-iterations"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Contest URL (optional)</label>
                    <input
                      type="url"
                      placeholder="https://www.draftkings.com/..."
                      className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                      id="quick-play-url"
                    />
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">Workflow Steps</h3>
                  <div className="space-y-3">
                    {[
                      "Fetch DraftKings slate data",
                      "Generate player projections",
                      "Create ownership estimates", 
                      "Run lineup optimizer",
                      "Simulate tournament results",
                      "Prepare final CSV for upload"
                    ].map((step, index) => (
                      <div key={index} className="flex items-center space-x-3 text-gray-300">
                        <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                          <span className="text-xs text-indigo-400">{index + 1}</span>
                        </div>
                        <span className="text-sm">{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col space-y-4">
                <button
                  onClick={() => handleRunCompleteWorkflow()}
                  disabled={isProcessing}
                  className="w-full bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-4 rounded-lg transition-colors flex items-center justify-center text-lg font-semibold"
                >
                  {isProcessing ? (
                    <>
                      <RefreshCw className="w-5 h-5 mr-3 animate-spin" />
                      Running Workflow...
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5 mr-3" />
                      Generate Lineups Now
                    </>
                  )}
                </button>
                
                <p className="text-xs text-gray-400 text-center">
                  This process typically takes 2-5 minutes to complete depending on slate size.
                </p>
              </div>
            </div>
            
            {/* Progress and Results */}
            {workflowProgress && workflowProgress.length > 0 && (
              <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
                <h3 className="text-xl font-bold text-white mb-4">Workflow Progress</h3>
                <div className="space-y-2">
                  {workflowProgress.map((step, index) => (
                    <div key={index} className={`text-sm ${step.startsWith('✓') ? 'text-green-400' : step.startsWith('✗') ? 'text-red-400' : 'text-gray-300'}`}>
                      {step}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {workflowResults && (
              <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                  <Download className="w-5 h-5 mr-2 text-green-400" />
                  Results Ready!
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-indigo-400">{workflowResults.slate_players}</div>
                    <div className="text-sm text-gray-400">Players</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">{workflowResults.projections}</div>
                    <div className="text-sm text-gray-400">Projections</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-400">{workflowResults.optimized_lineups}</div>
                    <div className="text-sm text-gray-400">Lineups</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-400">{workflowResults.simulation_iterations}</div>
                    <div className="text-sm text-gray-400">Iterations</div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
                  <button
                    onClick={() => handleDownloadResults()}
                    className="flex-1 bg-green-500 hover:bg-green-600 text-white px-4 py-3 rounded-lg transition-colors flex items-center justify-center font-semibold"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Lineups CSV
                  </button>
                  
                  <button
                    onClick={() => handleShowUploadInstructions()}
                    className="flex-1 bg-blue-500 hover:bg-blue-600 text-white px-4 py-3 rounded-lg transition-colors flex items-center justify-center font-semibold"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Instructions
                  </button>
                </div>
                
                {showUploadInstructions && (
                  <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <h4 className="font-semibold text-blue-400 mb-2">How to Upload to DraftKings:</h4>
                    <ol className="text-sm text-gray-300 space-y-1">
                      <li>1. Download the CSV file above</li>
                      <li>2. Go to your DraftKings contest</li>
                      <li>3. Click "Enter Multiple Lineups"</li>
                      <li>4. Upload the CSV file</li>
                      <li>5. Review and submit your entries</li>
                    </ol>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Web Scraper Tab */}
        {activeTab === 'web-scraper' && (
          <FirecrawlPanel />
        )}

        {/* Optimizer Tab */}
        {activeTab === 'optimizer' && (
          <div className="space-y-6">
            <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2 text-indigo-400" />
                Lineup Optimizer
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Number of Lineups</label>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    defaultValue="20"
                    className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Unique Players</label>
                  <input
                    type="number"
                    min="1"
                    max="9"
                    defaultValue="1"
                    className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    onClick={() => handleRunOptimizer()}
                    disabled={isProcessing}
                    className="w-full bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center"
                  >
                    {isProcessing ? (
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Target className="w-4 h-4 mr-2" />
                    )}
                    {isProcessing ? 'Running...' : 'Run Optimizer'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Simulations Tab */}
        {activeTab === 'simulations' && (
          <div className="space-y-6">
            <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-green-400" />
                GPP Simulator
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Field Size</label>
                  <input
                    type="number"
                    min="10"
                    max="10000"
                    defaultValue="100"
                    className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Iterations</label>
                  <input
                    type="number"
                    min="100"
                    max="10000"
                    defaultValue="1000"
                    className="w-full bg-white/5 border border-white/10 text-white px-3 py-2 rounded-lg focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    onClick={() => handleRunSimulation()}
                    disabled={isProcessing}
                    className="w-full bg-green-500 hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center"
                  >
                    {isProcessing ? (
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Activity className="w-4 h-4 mr-2" />
                    )}
                    {isProcessing ? 'Running...' : 'Run Simulation'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Projections Tab */}
        {activeTab === 'projections' && (
          <div className="space-y-6">
            <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-blue-400" />
                Player Projections
              </h2>
              {dashboardData?.projectionSample?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="text-left py-2 text-gray-300">Player</th>
                        <th className="text-left py-2 text-gray-300">Team</th>
                        <th className="text-left py-2 text-gray-300">Position</th>
                        <th className="text-left py-2 text-gray-300">FPTS</th>
                        <th className="text-left py-2 text-gray-300">Own%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dashboardData.projectionSample.map((player, idx) => (
                        <tr key={idx} className="border-b border-white/5">
                          <td className="py-2 text-white">{player.name}</td>
                          <td className="py-2 text-gray-300">{player.team}</td>
                          <td className="py-2 text-gray-300">{player.pos}</td>
                          <td className="py-2 text-indigo-400">{player.fpts}</td>
                          <td className="py-2 text-green-400">{player['own%']}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-gray-400">No projection data available. Upload a projections file to get started.</p>
              )}
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-purple-400" />
                Analytics & Insights
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white/5 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">Performance Trends</h3>
                  <p className="text-gray-400 text-sm">Track lineup performance over time and identify optimization patterns.</p>
                </div>
                <div className="bg-white/5 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">Correlation Analysis</h3>
                  <p className="text-gray-400 text-sm">Analyze player correlations and stack effectiveness.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Dashboard Tab - Original Content */}
        {activeTab === 'dashboard' && (
          <>
            {loading && (
              <div className="flex items-center justify-center bg-black/30 border border-white/10 rounded-xl py-12">
                <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
                <span className="ml-3 text-sm text-gray-300">Loading dashboard data...</span>
              </div>
            )}

            {error && !loading && (
              <div className="flex items-center space-x-3 bg-red-500/10 border border-red-500/30 text-red-200 px-4 py-3 rounded-xl">
                <AlertTriangle className="w-5 h-5" />
                <div>
                  <p className="font-semibold">Data unavailable</p>
                  <p className="text-sm text-red-200/80">{error}</p>
                </div>
              </div>
            )}

            {!loading && !error && dashboardData && (
          <div className="grid grid-cols-12 gap-6">
            <aside className="col-span-12 lg:col-span-3 space-y-6">
              <section className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                <h3 className="text-white font-semibold mb-4 flex items-center">
                  <Zap className="w-4 h-4 mr-2 text-yellow-400" />
                  Optimization Settings
                </h3>
                {Object.entries(sliderValues).map(([key, value]) => (
                  <div key={key} className="mb-4">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-gray-400 capitalize">{key}</span>
                      <span className="text-sm text-indigo-400">{Math.round(value * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={value}
                      onChange={(event) => handleSliderChange(key, parseFloat(event.target.value))}
                      className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, rgb(99 102 241) 0%, rgb(99 102 241) ${value * 100}%, rgb(255 255 255 / 0.1) ${value * 100}%, rgb(255 255 255 / 0.1) 100%)`,
                      }}
                    />
                  </div>
                ))}

                <div className="mt-6 pt-4 border-t border-white/10">
                  <button className="w-full py-2 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg hover:from-indigo-600 hover:to-purple-700 transition-all">
                    Save Preferences
                  </button>
                </div>
              </section>

              <section className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                <h3 className="text-white font-semibold mb-4">Quick Stats</h3>
                <div className="space-y-3">
                  {(dashboardData.quickStats || []).map((stat) => (
                    <div key={stat.label} className="flex justify-between">
                      <span className="text-gray-400 text-sm">{stat.label}</span>
                      <span className="text-white font-semibold">{stat.value}</span>
                    </div>
                  ))}
                  {projectionSample.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm text-gray-300 uppercase tracking-wide mb-2">Top Projections</h4>
                      <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                        {projectionSample.map((row, index) => (
                          <div key={`${row.name}-${index}`} className="flex justify-between text-sm text-gray-300 border-b border-white/5 pb-1">
                            <div>
                              <p className="font-medium text-white">{row.name}</p>
                              <p className="text-xs text-gray-400">{row.team} • {row.pos}</p>
                            </div>
                            <div className="text-right">
                              {'fpts' in row && <p className="text-indigo-300 font-semibold">{Number(row.fpts).toFixed(2)}</p>}
                              {'own' in row && <p className="text-xs text-gray-400">Own {Number(row.own || 0).toFixed(1)}%</p>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </section>

              <section className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                <h3 className="text-white font-semibold mb-4">File Updates</h3>
                <ul className="space-y-2 text-sm text-gray-300">
                  {Object.entries(dashboardData.lastUpdated || {}).map(([key, value]) => (
                    <li key={key} className="flex justify-between">
                      <span className="capitalize text-gray-400">{key}</span>
                      <span>{value || '—'}</span>
                    </li>
                  ))}
                </ul>
              </section>
            </aside>

            <main className="col-span-12 lg:col-span-9 space-y-6">
              <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                {(dashboardData.metrics || []).map((metric) => {
                  const Icon = metricIcons[metric.label] || TrendingUp;
                  return (
                    <div key={metric.label} className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4 hover:border-white/20 transition-all">
                      <div className="flex items-center justify-between mb-2">
                        <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                          <Icon className="w-5 h-5 text-white" />
                        </div>
                      </div>
                      <div className="text-2xl font-bold text-white">{metric.value}</div>
                      <div className="text-sm text-gray-400">{metric.label}</div>
                    </div>
                  );
                })}
              </section>

              <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-white font-semibold">Performance vs. Win%</h3>
                    <button
                      onClick={() => setExpandedCard(expandedCard === 'performance' ? null : 'performance')}
                      className="text-gray-400 hover:text-white"
                    >
                      {expandedCard === 'performance' ? <ChevronUp /> : <ChevronDown />}
                    </button>
                  </div>
                  <ResponsiveContainer width="100%" height={expandedCard === 'performance' ? 360 : 220}>
                    <AreaChart data={dashboardData.performanceChart || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="label" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                      <Legend />
                      <Area type="monotone" dataKey="projection" stroke="#6366F1" fill="#6366F1" fillOpacity={0.3} name="Projection" />
                      <Area type="monotone" dataKey="ceiling" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.3} name="Ceiling" />
                      <Area type="monotone" dataKey="winRate" stroke="#10B981" fill="#10B981" fillOpacity={0.2} name="Win %" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                  <h3 className="text-white font-semibold mb-4">Ownership Leverage</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={dashboardData.ownershipChart || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="player" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                      <Legend />
                      <Bar dataKey="projected" fill="#6366F1" name="Proj Own %" />
                      <Bar dataKey="simulated" fill="#8B5CF6" name="Sim Own %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                  <h3 className="text-white font-semibold mb-4">Stack Composition</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={compositionData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={4} dataKey="value">
                        {compositionData.map((entry, index) => (
                          <Cell key={`slice-${entry.name}-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} formatter={(value) => `${value}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="flex flex-wrap gap-2 mt-4">
                    {compositionData.map((item) => (
                      <div key={item.name} className="flex items-center space-x-2">
                        <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                        <span className="text-xs text-gray-400">{item.name}: {item.value}%</span>
                      </div>
                    ))}
                    {compositionData.length === 0 && <p className="text-sm text-gray-500">Run a simulation to populate stack data.</p>}
                  </div>
                </div>

                <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                  <h3 className="text-white font-semibold mb-4">Simulation Health</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <RadarChart data={dashboardData.radarChart || []}>
                      <PolarGrid stroke="#374151" />
                      <PolarAngleAxis dataKey="stat" stroke="#9CA3AF" />
                      <PolarRadiusAxis stroke="#374151" />
                      <Radar name="Score" dataKey="value" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.5} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 overflow-hidden">
                <div className="p-4 border-b border-white/10 flex items-center justify-between">
                  <h3 className="text-white font-semibold">Contest Snapshot</h3>
                  <span className="text-xs text-gray-400">Source: {dashboardData.files?.simulation || '—'}</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10 text-gray-400">
                        <th className="text-left px-4 py-3 font-medium">Contest</th>
                        <th className="text-right px-4 py-3 font-medium">Entries</th>
                        <th className="text-right px-4 py-3 font-medium">Top Score</th>
                        <th className="text-right px-4 py-3 font-medium">Avg Score</th>
                        <th className="text-right px-4 py-3 font-medium">Avg ROI%</th>
                        <th className="text-right px-4 py-3 font-medium">Avg Return</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(dashboardData.contestTable || []).map((contest) => (
                        <tr key={contest.name} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3 text-white">{contest.name}</td>
                          <td className="px-4 py-3 text-right text-gray-300">{contest.entries ? contest.entries.toLocaleString() : '—'}</td>
                          <td className="px-4 py-3 text-right text-gray-300">{contest.topScore ? contest.topScore.toFixed(2) : '—'}</td>
                          <td className="px-4 py-3 text-right text-gray-300">{contest.avgScore ? contest.avgScore.toFixed(2) : '—'}</td>
                          <td className="px-4 py-3 text-right text-gray-300">{contest.roi ? contest.roi.toFixed(2) : '—'}</td>
                          <td className="px-4 py-3 text-right text-green-400">{contest.avgReturn ? `$${contest.avgReturn.toFixed(2)}` : '—'}</td>
                        </tr>
                      ))}
                      {(dashboardData.contestTable || []).length === 0 && (
                        <tr>
                          <td colSpan={6} className="px-4 py-6 text-center text-gray-400">
                            Run the simulator to populate contest metrics.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-white font-semibold">Lineup Insights</h3>
                  <span className="text-xs text-gray-400">{viewMode === 'grid' ? 'Showing top win-rate lineups' : 'Showing player exposure'} </span>
                </div>

                {viewMode === 'grid' ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {(dashboardData.lineupPreview || []).map((lineup, idx) => (
                      <div key={`lineup-${idx}`} className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-xs uppercase tracking-wide text-gray-400">Lineup #{idx + 1}</span>
                          <span className="text-xs text-indigo-300">Win {lineup.winPct.toFixed(2)}%</span>
                        </div>
                        <ul className="space-y-1 text-sm text-gray-200">
                          {lineup.lineup.map((player, playerIdx) => (
                            <li key={`${player}-${playerIdx}`} className="border-b border-white/5 pb-1 last:border-b-0 last:pb-0">
                              {player}
                            </li>
                          ))}
                        </ul>
                        <div className="flex justify-between text-xs text-gray-300">
                          <span>Proj: {lineup.projection.toFixed(2)}</span>
                          <span>ROI: {lineup.roi.toFixed(2)}</span>
                        </div>
                      </div>
                    ))}
                    {(dashboardData.lineupPreview || []).length === 0 && (
                      <p className="text-sm text-gray-400">No simulated lineups yet. Run `python -m src.main dk sim ...` to create them.</p>
                    )}
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400">
                          <th className="text-left px-4 py-3 font-medium">Player</th>
                          <th className="text-right px-4 py-3 font-medium">Win%</th>
                          <th className="text-right px-4 py-3 font-medium">Top1%</th>
                          <th className="text-right px-4 py-3 font-medium">Cash%</th>
                          <th className="text-right px-4 py-3 font-medium">Sim Own%</th>
                          <th className="text-right px-4 py-3 font-medium">Proj Own%</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(dashboardData.playerExposure || []).map((player) => (
                          <tr key={player.player} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                            <td className="px-4 py-3 text-white">{player.player}</td>
                            <td className="px-4 py-3 text-right text-gray-300">{player.winPct.toFixed(2)}%</td>
                            <td className="px-4 py-3 text-right text-gray-300">{player.topPct.toFixed(2)}%</td>
                            <td className="px-4 py-3 text-right text-gray-300">{player.cashPct.toFixed(2)}%</td>
                            <td className="px-4 py-3 text-right text-gray-300">{player.simOwn.toFixed(2)}%</td>
                            <td className="px-4 py-3 text-right text-gray-300">{player.projOwn.toFixed(2)}%</td>
                          </tr>
                        ))}
                        {(dashboardData.playerExposure || []).length === 0 && (
                          <tr>
                            <td colSpan={6} className="px-4 py-6 text-center text-gray-400">
                              Simulate a contest to capture exposure metrics.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </main>
          </div>
            )}
          </>
        )}
      </div>

      {isRefreshing && (
        <div className="fixed bottom-4 right-4 flex items-center space-x-2 bg-indigo-600/90 text-white px-4 py-2 rounded-full shadow-lg">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="text-sm font-medium">Syncing latest CSV output…</span>
        </div>
      )}
    </div>
  );
};

export default SimulationDashboard;

