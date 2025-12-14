This directory contain network traffic captures for EVSE-A and EVSE-B under normal and attack conditions. The csv files contains first 1000 packets extracted from the pcap files. To extract more features or packets from the pcaps, you can expand the included python script, pcap2csv.py which uses the NFStream package. 

Device ID Information
---------------------
Device Type 			Name 		Role 						Interface 		MAC / IP Address
-----------------------		-----------	----------------------------------------	---------		--------------------
Grizzl-E Smart Connect 		EVSE-A		Charging Station (OCPP Client)			Wifi			oc:8b:95:09:c6:08
Raspberry Pi			EVSE-B		Charging Station (OCPP Client)			Wifi			dc:a6:32:c9:e5:5f
Raspberry Pi			EVSE-B		Charging Station (V2G / ISO15118 Server)	Eth0			dc:a6:32:c9:e5:5e
Raspberry Pi			EVCC		EV (V2G / ISO15118 Client)			Eth0			dc:a6:32:c9:e6:9f
Raspberry Pi			Local CSMS	Local OCPP Server				Wifi			dc:a6:32:c9:e5:3e
Raspberry Pi			Attacker	Attacker					Wifi			dc:a6:32:dc:25:d5
Kali Linux PC			Attacker	Attacker					Wifi			a8:6b:ad:1f:9b:e5
Remote Server			Remote CSMS	Remote OCPP Server				 -			IP : 162.159.140.98


*Note that even though the table shows two attacker devices, only one was used at any given moment. 