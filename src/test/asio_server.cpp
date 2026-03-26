#include <array>
#include <iostream>
#include "../asio/include/asio.hpp"

using asio::ip::tcp;
#define PORT 13

int main(int argc, char* argv[]) {
	try {
		asio::io_context context;
		// tcp::acceptor acceptor{context, tcp::endpoint(tcp::v4(), PORT)};
		auto address = asio::ip::make_address("127.0.0.1");
		tcp::acceptor acceptor{context, tcp::endpoint(address, PORT)};

		for (;;) {
			asio::ip::tcp::socket socket{context};
			acceptor.accept(socket);

			time_t now = time(0);
			std::string mesg("eheheheheh");
			asio::write(socket, asio::buffer(mesg));
		}
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
