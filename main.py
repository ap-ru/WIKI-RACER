from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from urllib.parse import urlparse
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, end_node, visited_nodes):
    plt.figure(figsize=(12, 8))

    # Filter the graph to only include nodes that have outgoing edges
    outgoing_nodes = [node for node in G.nodes() if len(list(G.successors(node))) > 0]

    # Create a subgraph with only the outgoing nodes
    H = G.subgraph(outgoing_nodes)

    # Ensure the final node is included, even if it has no outgoing edges
    if end_node not in H:
        H = H.copy()  # Make a copy of the subgraph
        H.add_node(end_node)

    # Use spring layout with adjusted parameters to spread nodes apart
    pos = nx.spring_layout(H, k=0.8, iterations=100)  # Adjust k as needed

    # Node size based on the number of outgoing edges
    node_size = [len(list(H.successors(n))) * 100 for n in H.nodes()]

    # Draw nodes with transparency
    node_colors = ['orange' if node == end_node else 'skyblue' for node in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=node_size, alpha=0.7, node_color=node_colors)

    # Draw edges
    nx.draw_networkx_edges(H, pos, alpha=0.5)

    # Show labels for nodes with the highest degree, plus the final node
    labels = {n: n for n in sorted(H.nodes(), key=lambda n: len(list(H.successors(n))), reverse=True)[:]}
    
    # Add label for the end node
    """ end_node_label = end_node[end_node.rfind("/wiki/") + len("/wiki/"):]
    labels[end_node] = end_node_label  # Ensure the end node is labeled """

    nx.draw_networkx_labels(H, pos, labels, font_size=6)

    # Special handling to draw the end node if it has no outgoing edges
    """ if end_node not in G.nodes():
        # Place the final node at a specific location if needed
        pos[end_node] = (0, 0)  # Adjust position as necessary
        nx.draw_networkx_nodes(H, pos, nodelist=[end_node], node_size=2000, node_color='orange', alpha=0.7)
 """
    plt.title('Wikipedia Navigation Graph (Filtered for Outgoing Nodes)', fontsize=15)
    plt.axis('off')  # Turn off the axis
    plt.show()


# calculate_similarity takes in 2 words
def calculate_similarity(word1, word2):
    # creates WordNet synset for each word
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    if not synsets1 or not synsets2:
        return 0  # Return 0 if no synsets found
    
    # Take Wu-Palmer Similarity score between 2 synsets
    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    # returns the similarity score between the 2 words
    return max_similarity

def navigate_to_similar_link(driver, end_keyword, visited, previousLinks, page_graph, full_graph):
    # Find all appropriate links on the page
    links = driver.find_elements(By.CSS_SELECTOR, '#mw-content-text p a:not(sup a):not(.reflist a)')
    current_links = []

    current_url = driver.current_url

    if current_url not in full_graph:
        full_graph[current_url] = []

    # Calculate similarity for every link on the page
    max_similarity = 0
    for link in links:
        anchor_text = link.text.strip()
        similarity = calculate_similarity(anchor_text, end_keyword)
        href = link.get_attribute('href')

        if href and urlparse(href).netloc == 'en.wikipedia.org':
            full_graph[current_url].append(href)
    
        if href and urlparse(href).netloc == 'en.wikipedia.org' and anchor_text not in visited:
            if similarity != 0:
                current_links.append([link, similarity])
                driver.execute_script("window.scrollBy(0, 2);")
                if len(current_links) > 30 and len(current_links) % 5 == 0:
                    driver.execute_script("window.scrollBy(0, 200);")
                if similarity >= max_similarity:
                    print(anchor_text, int(similarity * 100))
                    max_similarity = similarity

    # Sort the links by similarity score (highest first)
    current_links.sort(key=lambda x: x[1], reverse=True)

    # Backtrack if similarity decreases
    if previousLinks and current_links and current_links[0][1] < previousLinks[-1][1]:
        print(f"\n\n*** SIMILARITY DECREASED. BACKTRACKING. ***\n\n")
        driver.back()
        driver.execute_script("window.scrollTo(0, 0);")
        return True

    # Explore the highest similarity link
    if current_links:
        next_topic = current_links[0][0].text.strip()
        next_href = current_links[0][0].get_attribute('href')

        # Add to visited and graph
        visited.append(next_topic)
        page_graph[driver.current_url] = [next_href, current_links[0][1]]

        current_links[0][0].click()
        print(f"\n\n*** GOING TO NEW PAGE: {next_topic} -- {current_links[0][1]} ***\n\n")
        return True
    else:
        driver.back() 
        return True
    
def main(start_keyword, end_keyword):
    # Initialize Chrome WebDriver
    options = Options()
    options.add_experimental_option("excludeSwitches",["enable-automation"])

    driver = webdriver.Chrome(options=options)

    visited = []
    previousLinks = []
    page_graph = {}
    full_graph = defaultdict(list)

    # Start from the Wikipedia page for the start keyword
    start_url = f"https://en.wikipedia.org/wiki/{start_keyword}"
    driver.get(start_url)

    # Navigate to Wikipedia pages until the page for the end keyword is reached
    while True:
        cur = driver.current_url.strip()
        check = cur[cur.rfind("/wiki/") + len("/wiki/"):]

        print(f"{check}\n")

        # Check if we have reached the end keyword Wikipedia page
        if check.lower() == end_keyword.lower() or check.lower().startswith(end_keyword.lower() + "_"):
            end_time = time.time()
            for node, neighbor in page_graph.items():
                print(f"Node {node} rerouted to {neighbor}")
            print(f"\n\nReached the Wikipedia page for '{end_keyword}' in {end_time - start_time} seconds!")
            print(f"Clicked on {len(previousLinks)} links")

            # Create a NetworkX graph from the full_graph
            nx_graph = nx.DiGraph()  # Use DiGraph for directed edges

            # Add edges to the graph
            for node, neighbors in full_graph.items():
                for neighbor in neighbors:
                    nx_graph.add_edge(node, neighbor)

            # Visualize the constructed graph
            visualize_graph(nx_graph, driver.current_url, visited)
            break
        
        # if not, repeat the process
        res = navigate_to_similar_link(driver, end_keyword, visited, previousLinks, page_graph, full_graph)
        previousLinks.append((check, page_graph.get(cur, [None, 0])[1]))  # Append current similarity score

        if not res:
            print(f"No similar link found for '{end_keyword}'. Exiting...")
            break

    # Close the browser
    time.sleep(10)
    driver.quit()


if __name__ == "__main__":
    start_keyword = "Statue of Liberty"
    end_keyword = "Hinduism"
    start_time = time.time()
    main(start_keyword, end_keyword)