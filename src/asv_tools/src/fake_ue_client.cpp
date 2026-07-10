#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using json = nlohmann::json;

int main(int argc, char ** argv)
{
  const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
  const int port = argc > 2 ? std::stoi(argv[2]) : 8080;
  const std::string terminator = "__OD_END__";

  const int socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    std::cerr << "socket() failed: " << std::strerror(errno) << '\n';
    return 1;
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    std::cerr << "Invalid IPv4 address: " << host << '\n';
    return 1;
  }

  if (::connect(socket_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
    std::cerr << "connect() failed: " << std::strerror(errno) << '\n';
    return 1;
  }

  timeval timeout{};
  timeout.tv_sec = 0;
  timeout.tv_usec = 20000;
  (void)::setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  std::cout << "Connected to " << host << ':' << port << '\n';
  const auto start = std::chrono::steady_clock::now();
  std::string receive_buffer;

  while (true) {
    const double time_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();

    const json state = {
      {"Time", time_seconds},
      {"Delta_Time", 0.1},
      {"Surge_Velocity", 0.0},
      {"Angular_Velocity", 0.0},
      {"ASV_Location", {{"X", 0.0}, {"Y", 0.0}, {"Z", 0.0}}},
      {"Target_Location", {
        {"X", 500.0 + 100.0 * std::cos(0.2 * time_seconds)},
        {"Y", 200.0 + 100.0 * std::sin(0.2 * time_seconds)},
        {"Z", 0.0}}},
      {"ASV_Rotation", {{"Roll", 0.0}, {"Pitch", 0.0}, {"Yaw", 0.0}}},
      {"Camera_Capture", json::array()}
    };

    const std::string payload = state.dump() + terminator;
    if (::send(socket_fd, payload.data(), payload.size(), MSG_NOSIGNAL) <= 0) {
      std::cerr << "send() failed\n";
      break;
    }

    char chunk[8192];
    const ssize_t received = ::recv(socket_fd, chunk, sizeof(chunk), 0);
    if (received > 0) {
      receive_buffer.append(chunk, static_cast<std::size_t>(received));
      while (true) {
        const std::size_t end = receive_buffer.find(terminator);
        if (end == std::string::npos) {
          break;
        }
        std::cout << "Jetson -> UE5: " << receive_buffer.substr(0, end) << '\n';
        receive_buffer.erase(0, end + terminator.size());
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  ::close(socket_fd);
  return 0;
}
