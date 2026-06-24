using LinearAlgebra

# Helper structure to manage active segments
mutable struct Segment
    a::Int
    b::Int
    cost::Float64
    best_split::Int
    split_cost::Float64
    benefit::Float64
end

"""
Builds a Sparse Table for O(1) Range Minimum Queries (RMQ).
Time Complexity: O(N log N)
"""
function build_sparse_table(X::Vector{Float64})
    n = length(X)
    max_log = floor(Int, log2(n))
    st = Matrix{Float64}(undef, n, max_log + 1)
    st[:, 1] = X
    
    for j in 1:max_log
        stride = 1 << (j - 1)
        for i in 1:(n - (1 << j) + 1)
            st[i, j+1] = min(st[i, j], st[i + stride, j])
        end
    end
    
    log_table = zeros(Int, n)
    for i in 2:n
        log_table[i] = log_table[i >> 1] + 1
    end
    
    return st, log_table
end

"""
Queries the minimum value in the range [a, b] in O(1) time.
"""
@inline function query_min(a::Int, b::Int, st::Matrix{Float64}, log_table::Vector{Int})
    len = b - a + 1
    k = log_table[len]
    return min(st[a, k+1], st[b - (1 << k) + 1, k+1])
end

"""
Creates a Segment object, finding its internal optimal split point.
Time Complexity: O(b - a)
"""
function create_segment(a::Int, b::Int, P1::Vector{Float64}, P2::Vector{Float64}, st::Matrix{Float64}, log_table::Vector{Int})
    # O(1) Cost evaluator using algebraic expansion
    get_cost(l, r) = begin
        m = query_min(l, r, st, log_table)
        sum_x = P1[r+1] - P1[l]
        sum_x2 = P2[r+1] - P2[l]
        return sum_x2 - 2.0 * m * sum_x + (m^2) * (r - l + 1)
    end
    
    cost = get_cost(a, b)
    
    # Base case: Base segments of length 1 cannot be split
    if a == b
        return Segment(a, b, cost, a, cost, 0.0)
    end
    
    best_s = a
    min_split_cost = Inf
    
    # Scan for the best split point within this segment
    for s in a:(b-1)
        total = get_cost(a, s) + get_cost(s+1, b)
        if total < min_split_cost
            min_split_cost = total
            best_s = s
        end
    end
    
    benefit = cost - min_split_cost
    return Segment(a, b, cost, best_s, min_split_cost, benefit)
end

"""
Greedy Top-Down Time Series Segmentation into k blocks.
Overall Time Complexity: O(N log N + k * N)
"""
function greedy_time_series_segmentation(X::Vector{Float64}, k::Int)
    n = length(X)
    if k < 1 || k > n
        error("k must be between 1 and the length of the array.")
    end
    
    # 1. Precomputations (O(N log N))
    P1 = [0.0; cumsum(X)]
    P2 = [0.0; cumsum(X .^ 2)]
    st, log_table = build_sparse_table(X)
    
    # 2. Initialize pool with the entire array as a single segment
    initial_seg = create_segment(1, n, P1, P2, st, log_table)
    segments = [initial_seg]
    
    # 3. Iteratively split segments (k-1 times)
    for _ in 1:(k-1)
        best_idx = 0
        max_benefit = -1.0
        
        # Find the active segment whose split yields the biggest reduction in SSE
        for (idx, seg) in enumerate(segments)
            if seg.a < seg.b && seg.benefit > max_benefit
                max_benefit = seg.benefit
                best_idx = idx
            end
        end
        
        # If no further beneficial splits can be made, terminate early
        if best_idx == 0 || max_benefit <= 0.0
            break
        end
        
        # Pop the chosen segment
        seg_to_split = segments[best_idx]
        deleteat!(segments, best_idx)
        
        # Split it into two new segments and calculate their internal optimal metrics
        seg_left = create_segment(seg_to_split.a, seg_to_split.best_split, P1, P2, st, log_table)
        seg_right = create_segment(seg_to_split.best_split + 1, seg_to_split.b, P1, P2, st, log_table)
        
        push!(segments, seg_left)
        push!(segments, seg_right)
    end
    
    # 4. Format results chronologically
    sort!(segments, by = s -> s.a)
    
    intervals = [(seg.a, seg.b) for seg in segments]
    representatives = [query_min(seg.a, seg.b, st, log_table) for seg in segments]
    
    return intervals, representatives
end


function run_benchmark(k = 25)
    # ==========================================
    # Example Usage with Simulated Data
    # ==========================================

    # Generate a mock time series with 1,000,000 elements
    println("Generating 1,000,000 data points...")
    X = randn(1_000_000) 
    
    println("Running greedy aggregation into k = $k blocks...")
    # Benchmark execution time
    @time intervals, representatives = greedy_time_series_segmentation(X, k)
    
    println("\nResults:")
    for i in 1:k
        println("Block $i: Indices $(intervals[i]) -> Value (Minimum): $(representatives[i])")
    end
end

function aggregate(X, k::Int)
    println("Running greedy aggregation into k = $k blocks...")
    # Benchmark execution time
    @time intervals, representatives = greedy_time_series_segmentation(X, k)
    
    println("\nResults:")
    for i in 1:k
        println("Block $i: Indices $(intervals[i]) -> Value (Minimum): $(representatives[i])")
    end
end