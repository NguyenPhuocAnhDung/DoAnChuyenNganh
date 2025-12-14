This directory contains the CSV file for logged power consumption data for EVSE-B(rasperry pi) for all scenarios in this experiment. Each sample is timestamped and labelled. sampling rate is 1 sec. 

Feature			Description
------------------------------------------
Time			Timestamp of sample
Shunt_voltage (mV)	Voltage drop that occurs across a shunt resistor of I2C Wattmeter
Bus_voltage		DC Voltage supply
Current_mA		EVSE-B Current consumption
Power_mw		EVSE-B Power consumption
 

Labelling: 

Column ID			Entry
----------------------------------------------------
State				Idle, Charging
Scenario			Recon, DoS, Cryptojacking, Backdoor, Benign
Attack				Cryptojacking, Backdoor, None (ie. Benign), tcp-port-scan, service-version-detection, os-fingerpriting, aggressive-scan, syn-stealth-scan, 				vulnerability-scan, slowloris-scan, upd-flood, icmp-flood, pshack-flood, icmp-fragmentation, tcp-flood, syn-flood, synonymousIP-flood
Label				Attack, Benign
Interface			OCPP, ISO15118

