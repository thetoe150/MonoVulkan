#include <array>
#include <iostream>
#include "../asio/include/asio.hpp"

#define PORT 13

int main(int argc, char* argv[]) {
	try {
		if (argc < 2) 
			throw "Too few argument!!!\n";

		asio::io_context context;
		// host name or domain name
		// asio::ip::tcp::resolver resolver{context};
		// asio::ip::tcp::resolver::results_type endpoints = resolver.resolve(argv[1], "daytime");
		
		auto address = asio::ip::make_address("127.0.0.1");
		asio::ip::tcp::endpoint endpoint(address, PORT);
		asio::ip::tcp::socket socket{context};
		socket.connect(endpoint);

		for (;;) {
			std::array<char, 128> buf;
			std::error_code err;

			size_t len = socket.read_some(asio::buffer(buf), err);
			if (err == asio::error::eof)
				break; // Connection closed cleanly by peer.
			else if (err) {
				throw "Connection error!!!\n";
			}
			std::cout.write(buf.data(), len);
		}
	}
	catch (std::exception& e){
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
