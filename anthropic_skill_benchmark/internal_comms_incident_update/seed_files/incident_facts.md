# Incident Facts

Time:
- Queue buildup started at around 09:20
- Service recovered at around 09:45

Initial suspected cause:
- Worker auto-scaling did not trigger as expected

Impact:
- 12 internal experiment tasks were delayed
- No external users were affected
- There is no sign of data loss

Current status:
- The service has recovered
- The team is checking scaling trigger logic and alert thresholds
