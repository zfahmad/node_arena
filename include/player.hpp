#ifndef PLAYER_INDEXED_H
#define PLAYER_INDEXED_H

#include <array>

template <typename T, typename PlayerEnum> class PlayerIndexed {
public:
    constexpr PlayerIndexed() : data_{} {}

    constexpr explicit PlayerIndexed(const std::array<T, 2> &arr)
        : data_{arr} {}

    constexpr T &operator[](PlayerEnum p) noexcept {
        return data_[to_index(p)];
    }

    constexpr const T operator[](PlayerEnum p) const noexcept {
        return data_[to_index(p)];
    }

    constexpr const std::array<T, 2> &data() const noexcept { return data_; }

private:
    static constexpr std::size_t to_index(PlayerEnum p) noexcept {
        return static_cast<std::size_t>(p);
    }
    std::array<T, 2> data_;
};

#endif
