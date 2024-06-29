# Description: Example data and tasks for the data loader
tasks = {
    "translation": [
        {
            "prompt": "Translate English to French: Hello, how are you today?",
            "gt": "Bonjour, comment vas-tu aujourd'hui ?"
        },
        {
            "prompt": "Translate English to Spanish: What time is the meeting tomorrow?",
            "gt": "¿A qué hora es la reunión mañana?"
        },
        {
            "prompt": "Translate English to German: I would like a cup of coffee, please.",
            "gt": "Ich hätte gerne eine Tasse Kaffee, bitte."
        },
        {
            "prompt": "Translate English to Italian: It’s a beautiful day, isn’t it?",
            "gt": "È una bella giornata, non è vero?"
        },
        {
            "prompt": "Translate English to Japanese: Can you help me with this problem?",
            "gt": "この問題を手伝ってくれますか？"
        }
    ],
    "summarization": [
        {
            "prompt": "summarize: The Wright brothers, Orville and Wilbur, were two American pioneers of aviation who are credited with inventing and building the world's first successful motor-operated airplane. Through their company, they also made the first controlled, sustained flight of a powered, heavier-than-air aircraft on December 17, 1903, in Kitty Hawk, North Carolina. Beyond their initial invention, they developed their flying machine into the first practical fixed-wing aircraft. The brothers' fundamental breakthrough was their invention of three-axis control, which enabled the pilot to steer the aircraft effectively and to maintain its equilibrium.",
            "gt": "The Wright brothers invented and flew the first successful motor-operated airplane."
        },
        {
            "prompt": "summarize: Shakespeare's Romeo and Juliet is a timeless tragedy about two young star-crossed lovers whose untimely deaths ultimately reconcile their feuding families. Set in the city of Verona, the story explores themes of love, fate, and the consequences of family conflict. The narrative follows the intense romance between Romeo and Juliet, against the backdrop of a bitter feud between their families, the Montagues and the Capulets. Despite the lovers' efforts to be together, a series of unfortunate events leads to their tragic demise, sparking a moment of reconciliation among their warring families.",
            "gt": "Romeo and Juliet's deaths reconcile their feuding families in this tragic love story."
        },
        {
            "prompt": "summarize: Photosynthesis is a complex process that green plants, algae, and some bacteria use to turn light energy into chemical energy. This transformation allows them to synthesize glucose and other nutrients from carbon dioxide and water, releasing oxygen as a byproduct. The process primarily occurs in the chloroplasts of plant cells, where chlorophyll captures sunlight. Photosynthesis is crucial for life on Earth, as it forms the base of the food chain, provides oxygen, and absorbs carbon dioxide, helping to regulate the planet's atmosphere.",
            "gt": "Photosynthesis converts light into chemical energy, producing oxygen and nutrients."
        },
        {
            "prompt": "summarize: The discovery of penicillin by Alexander Fleming in 1928 marked the beginning of the modern antibiotic era. Penicillin, a group of antibiotics derived from Penicillium fungi, was the first true antibiotic to be used in fighting bacterial infections. Its discovery revolutionized medicine by significantly reducing the death rate from bacterial infections. Before penicillin, even minor infections could be fatal, and there were no effective treatments for diseases like pneumonia, gonorrhea, or rheumatic fever. Penicillin's effectiveness against a wide range of bacteria made it a cornerstone of modern antibiotics.",
            "gt": "Penicillin's discovery began the antibiotic era, drastically reducing bacterial infection deaths."
        },
        {
            "prompt": "summarize: World War I, also known as the Great War, was a global conflict that lasted from 1914 to 1918, involving most of the world's nations. The immediate cause was the assassination of Archduke Franz Ferdinand of Austria-Hungary. However, the true causes were deeper and included militarism, alliances, imperialism, and nationalism. These factors created a tense environment in Europe, eventually leading to a war that involved many of the world's powers. The conflict resulted in significant loss of life and changed the geopolitical landscape of the 20th century.",
            "gt": "World War I was caused by militarism, alliances, imperialism, and nationalism."
        }
    ]
}
