import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Zap,
  Clock,
  Activity,
  CheckCircle,
  XCircle,
  RotateCcw,
  Bell,
  DollarSign,
  Target
} from 'lucide-react';

const PropTracker = () => {
  const [trackerStatus, setTrackerStatus] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Auto-refresh tracker status
  useEffect(() => {
    fetchTrackerStatus();
    const interval = setInterval(fetchTrackerStatus, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchTrackerStatus = async () => {
    try {
      const response = await fetch('/api/prop-tracker/status');
      if (response.ok) {
        const data = await response.json();
        setTrackerStatus(data);
        setIsTracking(data.active);
        setLastUpdate(new Date());
      }
    } catch (err) {
      console.error('Failed to fetch tracker status:', err);
    }
  };

  const startTracking = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/prop-tracker/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interval_minutes: 5
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start tracking');
      }
      
      const data = await response.json();
      setTrackerStatus(data);
      setIsTracking(true);
      await fetchTrackerStatus(); // Refresh status
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runSingleCycle = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/prop-tracker/run-cycle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to run tracking cycle');
      }
      
      const data = await response.json();
      await fetchTrackerStatus(); // Refresh status
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const markAlertProcessed = async (alertId) => {
    try {
      const response = await fetch(`/api/prop-tracker/alert/${alertId}/mark-processed`, {
        method: 'POST'
      });
      
      if (response.ok) {
        await fetchTrackerStatus(); // Refresh to update alert status
      }
    } catch (err) {
      console.error('Failed to mark alert as processed:', err);
    }
  };

  const getAlertLevelColor = (level) => {
    switch (level) {
      case 'critical':
        return 'text-red-400 bg-red-500/20 border-red-500/30';
      case 'high':
        return 'text-orange-400 bg-orange-500/20 border-orange-500/30';
      case 'medium':
        return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'low':
        return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
      default:
        return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getAlertIcon = (type) => {
    switch (type) {
      case 'line_movement':
        return <TrendingUp className="w-4 h-4" />;
      case 'arbitrage':
        return <DollarSign className="w-4 h-4" />;
      default:
        return <Bell className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-black/30 backdrop-blur-lg rounded-xl border border-white/10 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Target className="w-6 h-6 text-green-400 mr-3" />
          <h2 className="text-xl font-bold text-white">Prop Tracker</h2>
          <span className="ml-2 px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">
            Real-time Monitoring
          </span>
        </div>
        
        <div className="flex items-center space-x-3">
          {lastUpdate && (
            <span className="text-xs text-gray-400">
              Last update: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
          <div className={`w-3 h-3 rounded-full ${isTracking ? 'bg-green-400' : 'bg-gray-400'}`} />
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex space-x-4 mb-6">
        <button
          onClick={startTracking}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-green-500 hover:bg-green-600 disabled:opacity-50 text-white rounded-lg transition-colors"
        >
          {loading ? <RotateCcw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          <span>Start Tracking</span>
        </button>

        <button
          onClick={runSingleCycle}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white rounded-lg transition-colors"
        >
          {loading ? <RotateCcw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
          <span>Run Cycle</span>
        </button>

        <button
          onClick={fetchTrackerStatus}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-gray-500 hover:bg-gray-600 disabled:opacity-50 text-white rounded-lg transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 flex items-start space-x-3 bg-red-500/10 border border-red-500/30 text-red-200 px-4 py-3 rounded-lg">
          <XCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-semibold">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Tracker Status */}
      {trackerStatus && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">Active Props</span>
                <Activity className="w-4 h-4 text-blue-400" />
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {trackerStatus.summary?.active_props || 0}
              </p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">Platforms</span>
                <Target className="w-4 h-4 text-green-400" />
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {trackerStatus.summary?.platforms_tracked || 0}
              </p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">Recent Moves</span>
                <TrendingUp className="w-4 h-4 text-orange-400" />
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {trackerStatus.summary?.recent_movements || 0}
              </p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-300">Opportunities</span>
                <DollarSign className="w-4 h-4 text-purple-400" />
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {trackerStatus.summary?.active_opportunities || 0}
              </p>
            </div>
          </div>

          {/* Recent Alerts */}
          {trackerStatus.recent_alerts && trackerStatus.recent_alerts.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Bell className="w-5 h-5 mr-2" />
                Recent Alerts ({trackerStatus.recent_alerts.length})
              </h3>
              
              <div className="space-y-3">
                {trackerStatus.recent_alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`border rounded-lg p-4 ${getAlertLevelColor(alert.level)} ${
                      alert.processed ? 'opacity-60' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-3">
                        <div className="mt-1">
                          {getAlertIcon(alert.type)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <span className="font-medium text-white">
                              {alert.player} - {alert.stat}
                            </span>
                            <span className={`px-2 py-1 text-xs rounded-full ${getAlertLevelColor(alert.level)}`}>
                              {alert.level}
                            </span>
                            <span className="text-xs text-gray-400 capitalize">
                              {alert.type.replace('_', ' ')}
                            </span>
                          </div>
                          <p className="text-sm text-gray-200">
                            {alert.message}
                          </p>
                          <p className="text-xs text-gray-400 mt-1">
                            {new Date(alert.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {alert.processed ? (
                          <CheckCircle className="w-5 h-5 text-green-400" />
                        ) : (
                          <button
                            onClick={() => markAlertProcessed(alert.id)}
                            className="px-3 py-1 bg-white/10 text-white text-xs rounded hover:bg-white/20 transition-colors"
                          >
                            Mark Done
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Status Message */}
          {!trackerStatus.active && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                <div>
                  <p className="text-yellow-200 font-medium">Tracker Not Active</p>
                  <p className="text-sm text-yellow-200/80">
                    {trackerStatus.message || 'Click "Start Tracking" to begin monitoring prop lines'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="mt-6 bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="text-blue-200 font-medium mb-2">Prop Tracker Features</h4>
        <ul className="text-sm text-blue-200/80 space-y-1">
          <li>• <strong>Real-time Monitoring:</strong> Tracks prop lines across Underdog and PrizePicks</li>
          <li>• <strong>Line Movement Alerts:</strong> Notifies when lines move significantly</li>
          <li>• <strong>Arbitrage Detection:</strong> Finds profitable opportunities across platforms</li>
          <li>• <strong>Value Tracking:</strong> Monitors +EV props and market inefficiencies</li>
          <li>• <strong>Automated Updates:</strong> Runs continuous cycles to stay current</li>
        </ul>
      </div>
    </div>
  );
};

export default PropTracker;