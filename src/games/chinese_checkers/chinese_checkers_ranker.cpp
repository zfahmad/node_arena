#include <bit>
#include <games/chinese_checkers/chinese_checkers_ranker.hpp>

Ranker::Ranker(int dim, int pieces)
    : dim_(dim), N_(dim * dim), pieces_(pieces),
      C_(static_cast<size_t>(N_ + 1) * (pieces_ + 1), 0) {
    build_table();
}

BBType Ranker::bit_to_compact(BBType bitboard) const {
    BBType dense = (uint128_t)0;
    BBType shifted = bitboard >> (MAX_ROW + 1);
    BBType row_mask = ((uint128_t)1 << dim_) - 1;
    for (int i = 0; i < dim_; i++) {
        BBType row_bits =
            (shifted >> (i * MAX_ROW)) & row_mask; // isolate row i
        dense |= row_bits << (i * dim_);           // pack tightly
    }
    return dense;
}

BBType Ranker::rank_combination(BBType dense) const {
    BBType r = 0;
    int k = 1;
    BBType bits = dense;
    while (bits) {
        int pos = std::countr_zero(bits);
        r += at(pos, k);
        bits &= bits - 1;
        k++;
    }
    return r;
}

BBType Ranker::unrank_combination(BBType rank) const {
    BBType result = 0;
    int k = pieces_;
    for (int c = N_ - 1; c >= 0 && k > 0; c--) {
        BBType val = at(c, k);
        if (val <= rank) {
            result |= ((uint128_t)1 << c);
            rank -= val;
            k--;
        }
    }
    return result;
}

BBType Ranker::relabel_after_removal(BBType bits, BBType removed) const {
    BBType result = 0;
    while (bits) {
        int pos = std::countr_zero(bits);
        int shift = std::popcount(removed & (((uint128_t)1 << pos) - 1));
        result |= (uint128_t)1 << (pos - shift);
        bits &= bits - 1;
    }
    return result;
}

BBType
Ranker::rank_canonical_state(std::array<BBType, 2> canonical_state) const {
    BBType player_dense = bit_to_compact(canonical_state[0]);
    BBType opponent_dense = bit_to_compact(canonical_state[1]);
    BBType rank1 = rank_combination(player_dense);
    BBType opp_relabeled = relabel_after_removal(opponent_dense, player_dense);
    BBType rank2 = rank_combination(opp_relabeled);
    BBType base = at(N_ - pieces_, pieces_);
    return rank1 * base + rank2;
}

void Ranker::build_table() {
    for (int n = 0; n <= N_; n++) {
        at(n, 0) = 1;
        for (int k = 1; k <= pieces_ && k <= n; k++) {
            at(n, k) = at(n - 1, k - 1) + at(n - 1, k);
        }
    }
}
