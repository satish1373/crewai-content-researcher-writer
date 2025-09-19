import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Button, DatePicker, Select, Spin } from 'antd';
import { WebSocket } from 'ws';
import 'antd/dist/antd.css';

const { Option } = Select;

const Dashboard = () => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [dateRange, setDateRange] = useState([null, null]);
    const [filter, setFilter] = useState("");

    useEffect(() => {
        const ws = new WebSocket('ws://your-websocket-endpoint');

        ws.onmessage = (event) => {
            const newData = JSON.parse(event.data);
            setData(prevData => [...prevData, newData]);
        };

        ws.onopen = () => {
            console.log('WebSocket connection established');
        };

        return () => ws.close();
    }, []);

    const handleFilterChange = (value) => {
        setFilter(value);
    };

    const fetchHistoricalData = async () => {
        setLoading(true);
        try {
            const response = await fetch(`API_ENDPOINT?start=${dateRange[0]}&end=${dateRange[1]}&filter=${filter}`);
            const historicalData = await response.json();
            setData(historicalData);
        } catch (error) {
            console.error('Error fetching historical data:', error);
        } finally {
            setLoading(false);
        }
    };

    const chartData = {
        labels: data.map(entry => entry.timestamp),
        datasets: [
            {
                label: 'Data Over Time',
                data: data.map(entry => entry.value),
                borderColor: 'rgba(75,192,192,1)',
                fill: false,
            },
        ],
    };

    return (
        <div>
            <h2>Responsive Dashboard</h2>
            <Select defaultValue="" onChange={handleFilterChange} style={{ width: 120 }}>
                <Option value="">All</Option>
                <Option value="filter1">Filter 1</Option>
                <Option value="filter2">Filter 2</Option>
            </Select>
            <DatePicker.RangePicker 
                onChange={(dates) => setDateRange(dates)} 
                style={{ marginLeft: '10px' }} 
            />
            <Button type="primary" onClick={fetchHistoricalData} style={{ marginLeft: '10px' }}>
                Fetch Data
            </Button>

            {loading ? <Spin /> : <Line data={chartData} />}
        </div>
    );
};

export default Dashboard;
