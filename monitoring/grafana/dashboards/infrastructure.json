{
  "uid": "legal-ai-infrastructure",
  "title": "Legal AI | Infrastructure Monitoring",
  "description": "Infrastructure observability for the Enterprise Legal Document Intelligence AI platform: compute, containers, Kubernetes, PostgreSQL, Redis, storage, networking, and monitoring health.",
  "tags": [
    "legal-ai",
    "infrastructure",
    "devops",
    "prometheus",
    "docker",
    "kubernetes",
    "postgresql",
    "redis"
  ],
  "timezone": "browser",
  "schemaVersion": 39,
  "version": 1,
  "refresh": "30s",
  "editable": true,
  "graphTooltip": 1,
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "fiscalYearStartMonth": 0,
  "links": [],
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": {
          "type": "grafana",
          "uid": "-- Grafana --"
        },
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "templating": {
    "list": [
      {
        "name": "datasource",
        "label": "Prometheus",
        "type": "datasource",
        "query": "prometheus",
        "refresh": 1,
        "current": {
          "text": "Prometheus",
          "value": "prometheus"
        }
      },
      {
        "name": "instance",
        "label": "Node Instance",
        "type": "query",
        "datasource": {
          "type": "prometheus",
          "uid": "${datasource}"
        },
        "query": {
          "query": "label_values(node_uname_info, instance)",
          "refId": "StandardVariableQuery"
        },
        "refresh": 2,
        "multi": true,
        "includeAll": true,
        "allValue": ".*",
        "current": {
          "text": "All",
          "value": "$__all"
        }
      },
      {
        "name": "namespace",
        "label": "Kubernetes Namespace",
        "type": "query",
        "datasource": {
          "type": "prometheus",
          "uid": "${datasource}"
        },
        "query": {
          "query": "label_values(kube_pod_info, namespace)",
          "refId": "StandardVariableQuery"
        },
        "refresh": 2,
        "multi": true,
        "includeAll": true,
        "allValue": ".*",
        "current": {
          "text": "All",
          "value": "$__all"
        }
      }
    ]
  },
  "panels": [
    {
      "id": 1,
      "type": "row",
      "title": "Infrastructure Overview",
      "gridPos": {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 0
      },
      "collapsed": false,
      "panels": []
    },
    {
      "id": 2,
      "type": "stat",
      "title": "Healthy Nodes",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(up{job=\"node-exporter\",instance=~\"${instance:regex}\"})",
          "legendFormat": "Healthy nodes"
        }
      ],
      "gridPos": {
        "h": 5,
        "w": 6,
        "x": 0,
        "y": 1
      },
      "options": {
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "orientation": "auto",
        "textMode": "auto",
        "colorMode": "value",
        "graphMode": "area"
      },
      "fieldConfig": {
        "defaults": {
          "min": 0,
          "decimals": 0,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "red",
                "value": null
              },
              {
                "color": "green",
                "value": 1
              }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "id": 3,
      "type": "stat",
      "title": "Average Node CPU Usage",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "1 - avg(rate(node_cpu_seconds_total{mode=\"idle\",instance=~\"${instance:regex}\"}[5m]))",
          "legendFormat": "CPU usage"
        }
      ],
      "gridPos": {
        "h": 5,
        "w": 6,
        "x": 6,
        "y": 1
      },
      "options": {
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "colorMode": "value",
        "graphMode": "area"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "orange",
                "value": 0.7
              },
              {
                "color": "red",
                "value": 0.9
              }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "id": 4,
      "type": "stat",
      "title": "Average Node Memory Usage",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "1 - (sum(node_memory_MemAvailable_bytes{instance=~\"${instance:regex}\"}) / clamp_min(sum(node_memory_MemTotal_bytes{instance=~\"${instance:regex}\"}),1))",
          "legendFormat": "Memory usage"
        }
      ],
      "gridPos": {
        "h": 5,
        "w": 6,
        "x": 12,
        "y": 1
      },
      "options": {
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "colorMode": "value",
        "graphMode": "area"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "orange",
                "value": 0.75
              },
              {
                "color": "red",
                "value": 0.9
              }
            ]
          }
        },
        "overrides": []
      }
    },
    {
      "id": 5,
      "type": "stat",
      "title": "Running Kubernetes Pods",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(kube_pod_status_phase{namespace=~\"${namespace:regex}\",phase=\"Running\"} == 1)",
          "legendFormat": "Running pods"
        }
      ],
      "gridPos": {
        "h": 5,
        "w": 6,
        "x": 18,
        "y": 1
      },
      "options": {
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "colorMode": "value",
        "graphMode": "area"
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0,
          "decimals": 0
        },
        "overrides": []
      }
    },
    {
      "id": 6,
      "type": "timeseries",
      "title": "Node CPU Usage by Instance",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "1 - avg by (instance)(rate(node_cpu_seconds_total{mode=\"idle\",instance=~\"${instance:regex}\"}[5m]))",
          "legendFormat": "{{instance}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 6
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      },
      "options": {
        "legend": {
          "displayMode": "list",
          "placement": "bottom"
        }
      }
    },
    {
      "id": 7,
      "type": "timeseries",
      "title": "Node Memory Usage by Instance",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "1 - (node_memory_MemAvailable_bytes{instance=~\"${instance:regex}\"} / node_memory_MemTotal_bytes{instance=~\"${instance:regex}\"})",
          "legendFormat": "{{instance}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 6
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 8,
      "type": "row",
      "title": "Storage & Networking",
      "gridPos": {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 14
      },
      "collapsed": false,
      "panels": []
    },
    {
      "id": 9,
      "type": "timeseries",
      "title": "Filesystem Utilization",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "1 - (node_filesystem_avail_bytes{instance=~\"${instance:regex}\",fstype!~\"tmpfs|overlay|squashfs\"} / node_filesystem_size_bytes{instance=~\"${instance:regex}\",fstype!~\"tmpfs|overlay|squashfs\"})",
          "legendFormat": "{{instance}} {{mountpoint}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 15
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 10,
      "type": "timeseries",
      "title": "Disk Read / Write Throughput",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (instance)(rate(node_disk_read_bytes_total{instance=~\"${instance:regex}\"}[5m]))",
          "legendFormat": "{{instance}} read"
        },
        {
          "refId": "B",
          "expr": "sum by (instance)(rate(node_disk_written_bytes_total{instance=~\"${instance:regex}\"}[5m]))",
          "legendFormat": "{{instance}} write"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 15
      },
      "fieldConfig": {
        "defaults": {
          "unit": "Bps",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 11,
      "type": "timeseries",
      "title": "Network Receive / Transmit",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (instance)(rate(node_network_receive_bytes_total{instance=~\"${instance:regex}\",device!~\"lo|veth.*|docker.*\"}[5m]))",
          "legendFormat": "{{instance}} receive"
        },
        {
          "refId": "B",
          "expr": "sum by (instance)(rate(node_network_transmit_bytes_total{instance=~\"${instance:regex}\",device!~\"lo|veth.*|docker.*\"}[5m]))",
          "legendFormat": "{{instance}} transmit"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 23
      },
      "fieldConfig": {
        "defaults": {
          "unit": "Bps",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 12,
      "type": "timeseries",
      "title": "Network Errors",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (instance)(rate(node_network_receive_errs_total{instance=~\"${instance:regex}\"}[5m]))",
          "legendFormat": "{{instance}} receive errors"
        },
        {
          "refId": "B",
          "expr": "sum by (instance)(rate(node_network_transmit_errs_total{instance=~\"${instance:regex}\"}[5m]))",
          "legendFormat": "{{instance}} transmit errors"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 23
      },
      "fieldConfig": {
        "defaults": {
          "unit": "ops",
          "min": 0,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 13,
      "type": "row",
      "title": "Containers & Kubernetes",
      "gridPos": {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 31
      },
      "collapsed": false,
      "panels": []
    },
    {
      "id": 14,
      "type": "timeseries",
      "title": "Container CPU Usage",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (namespace,pod)(rate(container_cpu_usage_seconds_total{namespace=~\"${namespace:regex}\",container!=\"\",container!=\"POD\"}[5m]))",
          "legendFormat": "{{namespace}} / {{pod}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 32
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 10,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 15,
      "type": "timeseries",
      "title": "Container Memory Working Set",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (namespace,pod)(container_memory_working_set_bytes{namespace=~\"${namespace:regex}\",container!=\"\",container!=\"POD\"})",
          "legendFormat": "{{namespace}} / {{pod}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 32
      },
      "fieldConfig": {
        "defaults": {
          "unit": "bytes",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 10,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 16,
      "type": "timeseries",
      "title": "Kubernetes Pod Restarts",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (namespace,pod)(increase(kube_pod_container_status_restarts_total{namespace=~\"${namespace:regex}\"}[1h]))",
          "legendFormat": "{{namespace}} / {{pod}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 40
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0,
          "custom": {
            "drawStyle": "bars",
            "fillOpacity": 25,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 17,
      "type": "timeseries",
      "title": "Kubernetes Pods by Phase",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (phase)(kube_pod_status_phase{namespace=~\"${namespace:regex}\"} == 1)",
          "legendFormat": "{{phase}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 40
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 18,
      "type": "row",
      "title": "PostgreSQL & Redis",
      "gridPos": {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 48
      },
      "collapsed": false,
      "panels": []
    },
    {
      "id": 19,
      "type": "timeseries",
      "title": "PostgreSQL Active Connections",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (datname)(pg_stat_activity_count{state=\"active\"})",
          "legendFormat": "{{datname}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 49
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 20,
      "type": "timeseries",
      "title": "PostgreSQL Transaction Rate",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(rate(pg_stat_database_xact_commit[5m]))",
          "legendFormat": "Commits / second"
        },
        {
          "refId": "B",
          "expr": "sum(rate(pg_stat_database_xact_rollback[5m]))",
          "legendFormat": "Rollbacks / second"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 49
      },
      "fieldConfig": {
        "defaults": {
          "unit": "ops",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 21,
      "type": "timeseries",
      "title": "Redis Memory Usage",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "redis_memory_used_bytes",
          "legendFormat": "{{instance}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 57
      },
      "fieldConfig": {
        "defaults": {
          "unit": "bytes",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 22,
      "type": "timeseries",
      "title": "Redis Connected Clients",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "redis_connected_clients",
          "legendFormat": "{{instance}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 57
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "min": 0,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 12,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 23,
      "type": "row",
      "title": "Monitoring Health",
      "gridPos": {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 65
      },
      "collapsed": false,
      "panels": []
    },
    {
      "id": 24,
      "type": "timeseries",
      "title": "Prometheus Target Availability",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "avg by (job)(up)",
          "legendFormat": "{{job}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 66
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 10,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    },
    {
      "id": 25,
      "type": "timeseries",
      "title": "Prometheus Scrape Duration",
      "datasource": {
        "type": "prometheus",
        "uid": "${datasource}"
      },
      "targets": [
        {
          "refId": "A",
          "expr": "max by (job)(scrape_duration_seconds)",
          "legendFormat": "{{job}}"
        }
      ],
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 66
      },
      "fieldConfig": {
        "defaults": {
          "unit": "s",
          "min": 0,
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 10,
            "showPoints": "never"
          }
        },
        "overrides": []
      }
    }
  ]
}