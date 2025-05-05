import json

PUNCTUATIONS = set("。？！…….!?")

def segment_text(content, target_length=512):
    """
    Segment the `content` field of a JSONL document according to the following requirements:

    1. Locate the punctuation mark closest to a length of 512 characters from the start of the segment (e.g., "。", "？", "！", "，"). Split the text at this punctuation point to create segments, ensuring that each segment is approximately 512 characters long.

    2. If the total text length is 1100 characters, and the final segment is too short, merge it with the previous segment. Then, find the nearest punctuation mark around the midpoint of the merged segment and split it into two roughly equal parts.

    3. Add a new field, `["segment_id"]`, to each segment. Label the segments of the \(i\)-th text sequentially as `i.1`, `i.2`, and so on.
    """
    segments = []
    start = 0
    while start < len(content):
        end = min(start + target_length, len(content))
        
        # jump out of the loop
        if end == len(content):
            segments.append(content[start:end])
            break

        # find the nearest punctuation around the target_length
        split_pos = end
        while split_pos > start and content[split_pos] not in PUNCTUATIONS:
            split_pos -= 1

        # truncate text if finding appropriate punctuation
        if split_pos > start:
            segments.append(content[start:split_pos + 1])
            start = split_pos + 1
        else:
            segments.append(content[start:end])
            start = end

    # adjust the last 2 chunks so that there won't be a too short chunk
    if len(segments) > 1 and len(segments[-1]) < target_length // 2:
        last = segments.pop()
        penultimate = segments.pop()
        combined = penultimate + last

        mid_point = len(combined) // 2
        split_pos = mid_point
        while split_pos > 0 and combined[split_pos] not in PUNCTUATIONS:
            split_pos -= 1

        if split_pos > 0:
            segments.append(combined[:split_pos + 1])
            segments.append(combined[split_pos + 1:])
        else:
            # 如果未找到标点，强制等长分割
            segments.append(combined[:mid_point])
            segments.append(combined[mid_point:])

    return segments

# main process
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for idx, line in enumerate(infile):
            data = json.loads(line)
            content = data.get("content", "")
            
            if content:
                segments = segment_text(content)
                for seg_idx, segment in enumerate(segments):
                    new_data = data.copy()
                    new_data["content"] = segment
                    new_data["segment_id"] = f"{idx}.{seg_idx}"
                    outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

# example
input_file = "./data/laws_data.jsonl"
output_file = "./data/laws_data_chunk.jsonl"
process_jsonl(input_file, output_file)
